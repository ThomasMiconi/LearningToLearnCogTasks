import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pdb

import scipy
from scipy import ndimage
from scipy import linalg

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

import numpy as np
from numpy import fft 

from scipy import io as spio
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(precision=5) 
np.set_printoptions(precision=5) 



# Specify the test task (and its logical negation, which is also withheld from the training set)
# TESTTASK = 'nand'; TESTTASKNEG = 'and'
TESTTASK = 'dms'; TESTTASKNEG = 'dnms'



LR =  1e-2         # Adam (evolutionary) LR. 
WDECAY =  3e-4 # Evolutionary weight decay parameter (for the Adam optimizer)
MUTATIONSIZE =  3 * .01 #  Std dev of the Gaussian mutations of the evolutionary algorithm

# ALPHAACTPEN =  3 * 3e-3
ALPHAACTPEN =  3 * 3 *  10 * 3e-3   # When squaring

NBGEN =  5000 # 1700 # 500      # Number of generations per run


N = 70  # Number of neurons in the RNN.



BS =  500 #  500 # 1000         # Batch size, i.e. population size for the evolutionary algorithm. 
assert BS % 2 == 0      # Should be even because of antithetic sampling.

# Same parameters as GR Yang:
TAU =  100  # Neuron membrane constant, in ms
DT = 20     # Duration of a timestep, in ms


# All the following times are in *timesteps*, not ms
T =  50      # Number of *timesteps* per trial
STIMTIME = 20       # Duration of stimulus input, total, *in timesteps* (not ms)
REWARDTIME = 10     # Duration of reward signal period
RESPONSETIME = 10   # Duration of responze period  
STARTRESPONSETIME = 25  # Timestep  at which response period starts
ENDRESPONSETIME = STARTRESPONSETIME + RESPONSETIME
STARTREWARDTIME = 36    # Timsestep at which reward is deliverd and reward signal starts
ENDREWARDTIME = STARTREWARDTIME + REWARDTIME
assert ENDREWARDTIME < T


MODULTYPE = 'EXTERNAL' # 'INTERNAL'

# JINIT = 1.5 #   Scale constant of initial network weights. See Section 2.7 in the MML paper.
# TAU_ET = 1000.0    # Time constant of the eligibility trace (in ms)
# PROBAMODUL =  .03 # .1 #       Probability of receiving a random perturbation, for each neuron, at each timestep.
# ALPHAMODUL =  1.0 # .5 #      Scale of the random perturbations
# ETA =   .1 *   .1  * .03  if MODULTYPE == 'INTERNAL' else .03 #             Learning rate for lifetime plasticity
# MULOSSTRACE = .9    #   Time constant for the trace of previous losses that serves as a baseline for neuromodulation
# MAXDW =  1e-2 #          Maximum delta-weight permissible (per time step) for lifetime plasticity 
# INITALPHA = .5 # 0.0 #  .5 #        Initial alpha (plasticity parameter) value


JINIT = 1.5 #   Scale constant of initial network weights. See Section 2.7 in the MML paper.
TAU_ET = 1000.0    # Time constant of the eligibility trace (in ms)
PROBAMODUL =  .1 #       Probability of receiving a random perturbation, for each neuron, at each timestep.
ALPHAMODUL =   .5 #      Scale of the random perturbations
ETA =   .1 *   .1  * .03  if MODULTYPE == 'INTERNAL' else .03 #             Learning rate for lifetime plasticity
MULOSSTRACE = .9    #   Time constant for the trace of previous losses that serves as a baseline for external neuromodulation
MAXDW =  1e-2 #          Maximum delta-weight permissible (per time step) for lifetime plasticity 
INITALPHA = .5 # 0.0 #  .5 #        Initial alpha (plasticity parameter) value



# The name of all the tasks. 14 tasks in total, because "always respond 0" and "always respond 1" are not included.
alltasks = ['and', 'nand' , '01', 'anti01' , '10', 'anti10', 'watchstim1', 'watchstim2' ,'dms',  'antiwatchstim2', 'antiwatchstim1', 'or', 'nor', 'dnms']



NBSTIMNEURONS = 2   # 2 Stimulus neurons. Stimuli are binary, so both neurons receive opposite-valued inputs (or 0)
NBREWARDNEURONS = 2 # 6 # 2 # reward signal for this trial. A  value is represented with 2 inputs, as it is for stimulus neurons.
NBBIASNEURONS = 1   # Bias neurons. Activations clamped to BIASVALUE.
NBINPUTNEURONS = NBSTIMNEURONS + NBREWARDNEURONS  + NBBIASNEURONS    # The first NBINPUTS neurons in the network are neurons (includes the bias, noise and reward inputs)
NBRESPNEURONS = 2  # Response neurons for 0 and 1.
NBMODNEURONS = 2    # Neuromodulatory output neurons
NBOUTPUTNEURONS = NBRESPNEURONS  + NBMODNEURONS   # The last NBOUTPUTNEURONS neurons in the network are output neurons. Response neurons + Modulatory neuron.
NBRESPSIGNALNEURONS = NBRESPNEURONS     # Neurons that receive the response-given signal ("what response did I just give?")
STIMNEURONS = np.arange(NBSTIMNEURONS)
INPUTNEURONS = np.arange(NBINPUTNEURONS)
OUTPUTNEURONS = np.arange(N-NBOUTPUTNEURONS, N)
MODNEURONS = np.arange(N-NBOUTPUTNEURONS, N-NBOUTPUTNEURONS + NBMODNEURONS)
# NUMMODNEURON = N - NBOUTPUTNEURONS      # The modulatory neuron is the first output neuron
RESPNEURONS = np.arange(N-NBOUTPUTNEURONS+NBMODNEURONS, N) # Then come the response neurons
REWARDNEURONS = np.arange(NBSTIMNEURONS, NBSTIMNEURONS+NBREWARDNEURONS) # The neurons receiving (and broadcasting) the "reward for this trial" signal are the ones just after the stimulus inputs.
BIASNEURONS = np.arange(NBSTIMNEURONS+NBREWARDNEURONS, NBSTIMNEURONS+NBREWARDNEURONS+NBBIASNEURONS)
FIRSTRESPSIGNALNEURON = NBSTIMNEURONS+NBREWARDNEURONS+NBBIASNEURONS   # The first neuron that receives the response-given signal. We'll need this later
assert FIRSTRESPSIGNALNEURON == NBINPUTNEURONS
assert len(RESPNEURONS) == NBRESPNEURONS
RESPSIGNALNEURONS = np.arange(FIRSTRESPSIGNALNEURON, FIRSTRESPSIGNALNEURON +NBRESPSIGNALNEURONS)


BIASVALUE = 1.0



NBTASKSPERGEN = 1 # 2 #  2 task blocks per generation


# NBTRIALSLOSS = 70              # Evolutionary loss is evaluated over the last 100 trials of each block
# NBTRIALS =  50 + NBTRIALSLOSS  # Total number of trials per block
NBTRIALSLOSS = 100              # Evolutionary loss is evaluated over the last 100 trials of each block
NBTRIALS =  300 + NBTRIALSLOSS  # Total number of trials per block



REWARDSIZE = 3.0 #  3 * 3.0 # Size of the binary-reward signal (correct/incorrect)
STIMSIZE = 3.0 # Size of the stimulus input
RESPSIGNALSIZE = 3.0 # Size of the response-given signal


totalnbtasks = 0
ticstart = time.time()


# EVALW is to assess the behavior of an evolved network. Run it on a single batch of all tasks, without any mutation
EVALW = False
if EVALW:
    NBGEN = 1
    NBTASKSPERGEN = 1
    BS = 500
    MUTATIONSIZE = 0
    allresps=[]
    allstims=[]
    alltgts=[]




with torch.no_grad():  # We don't need PyTorch to keep track of gradients, since we're computing the gradient outselves (through evolution).
 
    PRINTING = True # if numgen == 0 or np.random.rand() < .05 else False

    # Initialize innate weights values
    w =  torch.randn(N,N)  * JINIT / np.sqrt(N) 
    w = w.to(device)
    
    # Initialize alpha values - the plasticity parameters (capital-pi in the paper)
    alpha = INITALPHA * torch.ones_like(w).to(device)

    # We zero out input weights to input neurons, though it probably doesn't have any effect.
    w.data[:NBINPUTNEURONS, :] = 0   # Each *row* of w contains the weights to a single neuron.
    # We also zero out the weights to neuromodulatory neurons, which probably does have an effect!
    w.data[MODNEURONS, :] = 0   # Each *row* of w contains the weights to a single neuron.
    winit = w.clone()

    # We will be using the Adam optimizer to apply our (hand-computed) evolutionary gradients
    optimizer = optim.Adam([w, alpha], lr=LR, weight_decay=WDECAY)  # Default betas=(0.9, 0.999)

    # Evolosses are real-valued losses used for evolution. Binarylosses are binary 'correct/wrong' signals, also used for logging.
    evolosses = []
    responses0 = []
    binarylosses = []
    wgradnorms = []
    mytaskprev = mytaskprevprev = mytaskprevprevprev = -1
    

    if EVALW and False:
        w = np.loadtxt('w.txt')
        w = torch.from_numpy(w).float().to(device)
        winit = w.clone()

        alpha = np.loadtxt('alpha.txt')
        alpha = torch.from_numpy(alpha).float().to(device)


    print("MODULTYPE is:",  MODULTYPE)
    assert MODULTYPE == 'EXTERNAL' or MODULTYPE == 'INTERNAL', "Modulation type must be 'INTERNAL' or 'EXTERNAL'"


    # Ready to start the evolutionary loop, iterating over generations (i.e. lifetimes). 

    for numgen in range(NBGEN):



        if numgen == NBGEN // 2:
            for param_group in optimizer.param_groups:
                param_group['lr']  /=  5.0



        # Every 10th generation is for testing on the withheld task (with no weight change)
        TESTING = False
        if numgen == 0 or numgen == NBGEN-1 or numgen % 10 == 0:
            TESTING = True
            if PRINTING:
                print("TESTING")
        if EVALW:
            TESTING = False


        tic = time.time()   
        responses0thisgen = []

        
        
        alpha.clip_(min=0)
    


        # Generating the population of mutated individuals:

        # First, batch the weights.
        bw = torch.dstack(BS*[w]).movedim(2,0).to(device)     # batched weights
        balpha = torch.dstack(BS*[alpha]).movedim(2,0).to(device)     # batched alphas
        # Generate the mutations, for both w and alpha
        # NOTE: batch element 0 (and BS/2, its antithetic pair) are NOT mutated, represent the curent unmutated candidate genotype.
        mutations_wandalpha = []
        for  n, x in enumerate( (bw, balpha) ):
            mutations = torch.randn_like(x, requires_grad=False).to(device) *  MUTATIONSIZE
            mutations[0,:,:] = 0  # 1st item in batch = current candidate
            mutations[BS//2:, :, :] = -mutations[:BS//2, :, :]    # Antithetic sampling for mutations ! Really helps.
            if TESTING or EVALW:
                mutations *= 0.0  # No mutation - results in batch score variance being caused only by randomness in trial order and (possibly) lifetime perturbations
            x += mutations  
            mutations_wandalpha.append(mutations)


        
        bw.data[:, :NBINPUTNEURONS, :] = 0  # Input neurons receive 0 connections. Probably not necessary.
        bworig = bw.clone()                 # Storing the weights for comparison purposes at the gradient step (below).

        lifelosses = torch.zeros(BS, requires_grad=False).to(device)
        lifemselosses = torch.zeros(BS, requires_grad=False).to(device)
        lifeactpens = torch.zeros(BS, requires_grad=False).to(device)
        lifeblosses = torch.zeros(BS, requires_grad=False).to(device)

        
        

        # Lifetime loop, iterates over task-blocks:
        for numtask in range(NBTASKSPERGEN):
            totalnbtasks += 1

            COLLECTMODOUTSANDREWINS = not EVALW and ( (numtask + numgen * 2) % 7 == 0  )

            # bpw = batched plastic weights
            bpw = torch.zeros_like(bw).to(device)  # For now, plastic weights are initialized to 0 at the beginning of each task.

            # Initialize neural states
            bstates = .1 * torch.ones(BS, N).to(device)  # bstates (batched states) contains the neural activations (before nonlinearity). Dimensionality appropriate for batched matrix multiplication. 
            bstates[:, INPUTNEURONS] = 0
            bresps = 1.0 * bstates  # bresps is the actual neural responses, after nonlinearity, and also serves as the input for the next step.
            bresps[:, BIASNEURONS] = BIASVALUE

            meanlosstrace = torch.zeros(BS, 2 * 2).to(device)
            bls = []    # Will store binary losses of all batch elements, for each trial of this task
            bl0s = []   # Same but only for batch element 0 (i.e. the unmutated candidate genome)
            ml0s = []   # MSE loss (the one used for evolution) for element 0 (unmutated candidate), of all trials for this task



            # Choose the task ! If not testing, makes sure it's different from recently chosen tasks.


            if TESTING: 
                mytask = TESTTASK 
                mytasknum = alltasks.index(mytask)
            else:
                while True:
                    mytasknum = np.random.randint(len(alltasks))

                    mytask = alltasks[mytasknum]

                    if ( (mytask!= TESTTASK)  
                        and (mytask != TESTTASKNEG)  # We withhold both the test task and its logical negation
                        and     (mytask != mytaskprev) 
                            and (mytask != mytaskprevprev) 
                    ):

                        break

                mytaskprevprev = mytaskprev; mytaskprev= mytask



            # # Only use AND and NAND as tasks
            # mytasknum = numtask % 4
            # mytask = alltasks[mytasknum]
            # mytaskprevprev = mytaskprev; mytaskprev= mytask


            btasks = []  # Tasks for the whole batch
            for ii in range(BS//2):
                if TESTING: 
                    cand_task = TESTTASK
                    cand_tasknum = alltasks.index(TESTTASK)
                else:
                    while True:
                        cand_tasknum = np.random.randint(len(alltasks))
                        cand_task = alltasks[cand_tasknum]
                        if ( (cand_task!= TESTTASK)  
                            and (cand_task != TESTTASKNEG)  # We withhold both the test task and its logical negation


                            and (cand_tasknum % 2 == (numgen // 2) % 2)  # Training on alternate halves of the training set at successive (pairs of) generations
                            # and (cand_tasknum % 4 == numgen  % 4)  # Training on alternate quarters of the training set at successive generations


                            ): 
                            break
                btasks.append(cand_task)

            btasks = btasks * 2     # Duplicating the list, so each antithetic pair has the same tasks. 



            if EVALW:
                btasks = [TESTTASK] * BS
                # btasks = alltasks * (BS // len(alltasks) + 1)
                # btasks = btasks[:BS]
                # with open('btasks.txt', 'w') as f:
                #     for item in btasks:
                #         f.write("%s\n" % item)


            # btasks = [mytask] * BS


            assert(len(btasks) == BS)


            # Cumulative MSE and binary losses for this task, over the last NBLOSSTRIALS of the block:
            taskmselosses = torch.zeros_like(lifemselosses).to(device)
            taskblosses = torch.zeros_like(lifemselosses).to(device)

            respz = []      # Response neuron outputs
            stimz = []      # Stimulus neurons  outputs
            modouts = []    # Neuromodulatory output
            rewins = []     # Received rewards (reward neuron outputs)


            if PRINTING:
                print("task[0]:", btasks[0], "task[1]:", btasks[1])
            
            # OK, ready to start the task.

            # Generate the task data (inputs and targets) for all trials:
            # taskdata = generateInputsAndTargetsForTask(mytask=mytask)

            eligtraces =   torch.zeros_like(bw, requires_grad=False).to(device)  # Initialize the eligibility traces at the start of each block/task.


            # Task loop, iterating over trials
            # You do NOT erase memory (neural activations or plastic weights) between successive trials ! 
            for numtrial in range(NBTRIALS):


                # # Akshully do initialize network activations for each trial - THIS IS ONLY FOR DEBUGGING / SIMPLER TEST TASK!
                # bresps.fill_(0)
                # bstates.fill_(0)

                # # We reinitialize only modulatory neuron activations for each trial - THIS IS ONLY FOR DEBUGGING / SIMPLER TEST TASK!
                # bresps[:, MODNEURONS] = 0
                # bstates[:, MODNEURONS] = 0


                
                # Initializations
                mselossesthistrial = torch.zeros(BS, requires_grad=False).to(device)     # MSE losses for this trial
                totalresps = torch.zeros(BS, NBRESPNEURONS, requires_grad=False).to(device)     # Will accumulate the total outputs of each network over the trial, so we can compute the network's response for this trial.  

                # Generate the inputs and targets for this trial:

                # Pick stimulus 1 and stimulus 2 for this trial (and for each batch member):
                stims1 = (torch.rand(BS, 1) > .5).float()
                stims2 = (torch.rand(BS, 1) > .5).float()




                # Antithetic pairs share the exact same stimuli
                stims1[BS//2:, :] = stims1[:BS//2, :]
                stims2[BS//2:, :] = stims2[:BS//2, :]



                # Actual temporal inputs:
                inpts = np.zeros((BS, NBSTIMNEURONS, STIMTIME)) 
                StimDur = STIMTIME 
                StartStim = 0
                # The two stimuli are presented in succession, with both input neurons locked in opposite values to each other:
                inpts[:, 0, StartStim:StartStim+StimDur//2 - 2] = 2.0 * stims1 - 1.0
                inpts[:, 0, StartStim+StimDur//2:StartStim+StimDur - 2] = 2.0 * stims2 - 1.0
                inpts[:, 1, StartStim:StartStim+StimDur] = -inpts[:, 0, StartStim:StartStim+StimDur] 

                inputs = torch.from_numpy(inpts).float().to(device)


            
                # Now we compute the targets, that is, the expected values of the output neurons, depending on inputs and tasks
                tgts = -100 * np.ones((BS, NBRESPNEURONS, RESPONSETIME)) 
          
                for ii in range(BS):
                    # First we generate the expected output for the non-null response neuron, based on inputs and task:
                    if btasks[ii] == 'watchstim1':
                        tgts[ii, 1, :] = stims1[ii, 0]
                    elif btasks[ii] == 'watchstim2':
                        tgts[ii, 1, :] = stims2[ii, 0]
                    elif btasks[ii] == 'antiwatchstim1':
                        tgts[ii, 1, :] = 1.0 - stims1[ii, 0]
                    elif btasks[ii] == 'antiwatchstim2':
                        tgts[ii, 1, :] = 1.0 - stims2[ii, 0]
                    elif btasks[ii] == 'and':
                        tgts[ii, 1, :] = (stims1[ii, 0] * stims2[ii, 0])
                    elif btasks[ii] == 'nand':
                        tgts[ii, 1, :] = 1.0 - (stims1[ii, 0] * stims2[ii, 0])
                    # These two lines add  25% running time to the entire program! looks like np.clip is *slow*.
                    # elif btasks[ii] == 'or':
                    #     tgts[ii, 1, :] = np.clip(stims1[ii, 0] + stims2[ii, 0], 0.0, 1.0)
                    # elif btasks[ii] == 'nor':
                    #     tgts[ii, 1, :] = 1.0 - np.clip(stims1[ii, 0] + stims2[ii, 0], 0.0, 1.0)
                    # Instead, we will clip after the full array is done. This should still work out the same.
                    elif btasks[ii] == 'or':
                        tgts[ii, 1, :] = stims1[ii, 0] + stims2[ii, 0]
                    elif btasks[ii] == 'nor':
                        tgts[ii, 1, :] = 1.0 - stims1[ii, 0] + stims2[ii, 0]
                    elif btasks[ii] == '10':
                        tgts[ii, 1, :] = stims1[ii, 0] * (1.0 - stims2[ii, 0])
                    elif btasks[ii] == 'anti10':
                        tgts[ii, 1, :] = stims1[ii, 0] * (1.0 - stims2[ii, 0])
                    elif btasks[ii] == '01':
                        tgts[ii, 1, :] = (1.0 - stims1[ii, 0]) * stims2[ii, 0]
                    elif btasks[ii] == 'anti01':
                        tgts[ii, 1, :] = 1.0 - (1.0 - stims1[ii, 0]) * stims2[ii, 0]
                    elif btasks[ii] == 'dms':
                        tgts[ii, 1, :] = (stims1[ii, 0]  == stims2[ii, 0])
                    elif btasks[ii] == 'dnms':
                        tgts[ii, 1, :] = (stims1[ii, 0]  != stims2[ii, 0])
                    else:
                        tgts[ii, 1, :] = (stims1[ii, 0]  == stims2[ii, 0])
                    
                tgts[:, 1, :] = np.clip(tgts[:, 1, :], 0.0, 1.0)


                # tgts[:, 1, :] = 1.0

                # The null-response neuron's expected output is just the opposite of the non-null response neuron output (response is either 0 or 1).
                tgts[:, 0, :] = 1.0 - tgts[:, 1, :]

                assert np.all(np.logical_or(tgts == 0.0 , tgts == 1.0))

                if EVALW:
                    alltgts.append(tgts[:,1, 0])
                    allstims.append(np.hstack((stims1, stims2)))

                # assert numgen < 2 or numtrial < 15



                targets = torch.from_numpy(tgts).float().to(device)     

                # In practice, we clip targets to 0.1/0.9 instead of actually 0.0/1.0. This may or may not help.
                targets.clip_(min=0.1, max=0.9)



                # raise ValueError


                # Run the network. Trial loop, iterating over timesteps
                for numstep in range(T):          

                    # Update neural activations, using previous-step bresps (actual neural outputs) as input:
                    bstates += (DT / TAU) * (-bstates +  torch.bmm((bw + balpha * bpw), bresps[:, :, None])[:,:,0] )  


                    # Applying the random perturbations on neural activations, both for noise and for the lifetime plasticity algorithm (node-perturbation)
                    # And also updating the eligibility trace appropriately
                    if numstep > 1 : 
                        perturbindices =  (torch.rand(1, N) < PROBAMODUL).int()   # Which neurons get perturbed?

                        # perturbindices[0, MODNEURONS] = 0 # We disable perturbations on neuromodulatory neurons for debugging...


                        perturbations = (ALPHAMODUL * perturbindices * (2 * torch.rand(1, N) - 1.0)).to(device)  # Note the dimensions: the same noise vector is applied to all elements in the batch (to save time!)
                        



                        if numtrial > NBTRIALS - 20:
                            perturbations.fill_(0)



                        bstates += perturbations
                        
                        # Node-perturbation: Hebbian eligibility trace = product between inputs (bresps from previous time step) and *perturbations* in outputs. dH = X * deltaY 
                        # We do this with a (batched) outer product between the (column) vector of perturbations (1 per neuron) and the (row) vector of inputs
                        # Note that here, since we have an RNN, the input is bresps - the network's responses from the previous time step
                        if torch.sum(perturbindices) > 0:
                            eligtraces += torch.bmm( perturbations.expand(BS, -1)[:, :, None],  bresps[:, None, :] ) 

                    # Eligibility traces, unlike actual plastic weights, are decaying
                    eligtraces -=  (DT / TAU_ET) * eligtraces


                    # We can now compute the actual neural responses for this time step, applying the appropriate nonlinearity to each neuron
                    bresps = bstates.clone() # F.leaky_relu(bstates)
                    # The following assumes that response neurons are the last neurons of the network !                        
                    bresps[:,N-NBRESPNEURONS:].sigmoid_()     # The response neurons (NOT output neurons - modulatory neuron not included!) are sigmoids, all others are tanh. An arbitrary design choice.
                    bresps[:,:N-NBRESPNEURONS].tanh_()
                    

                    # Are we in the input presentation period? Then apply the inputs.
                    # Inputs are clamping, fixing the response of the input neurons.
                    if numstep < STIMTIME:
                        # bresps[:, STIMNEURONS] = STIMSIZE * inputs[:, :, numstep]        
                        bresps[:, STIMNEURONS[0]:STIMNEURONS[-1]+1] = STIMSIZE * inputs[:, :, numstep]        
                    else:
                        bresps[:, STIMNEURONS[0]:STIMNEURONS[-1]+1] = 0
                        # bresps[:, STIMNEURONS] = 0

                    # Bias input is always-on, always clamping.
                    # bresps[:, BIASNEURONS] = BIASVALUE
                    bresps[:, BIASNEURONS[0]] = BIASVALUE

                    # All the responses have now been computed  for this step

                    # Are we in the response period? Then collect network response.
                    if numstep >= STARTRESPONSETIME and numstep < ENDRESPONSETIME:

                        assert numstep < STARTREWARDTIME
                        # Accumulate the total activation of each output neuron, so that we can compute the network's actual response at the end of response period:
                        # totalresps +=  bresps[:, RESPNEURONS] 
                        totalresps +=  bresps[:, RESPNEURONS[0]:RESPNEURONS[-1]+1] 
                        # Accumulate the MSE error between actual and expected outputs:
                        # mselossesthistrial += torch.sum( (bresps[:, RESPNEURONS] - targets[:, :, numstep - STARTRESPONSETIME]) ** 2, axis=1 ) / RESPONSETIME
                        mselossesthistrial += torch.sum( (bresps[:, RESPNEURONS[0]:RESPNEURONS[-1]+1] - targets[:, :, numstep - STARTRESPONSETIME]) ** 2, axis=1 ) / RESPONSETIME

                    else:
                        bresps[:, RESPNEURONS[0]:RESPNEURONS[-1]+1] = 0.0
                        # bresps[:, RESPNEURONS] = 0.0


                    # Is the response period for this trial finished, or equivalently, are we at the first step of the reward / feedback period?
                    # If so, compute the network's response (i.e. which neuron fired most)
                    # Also, if using external neuromodulation, with compute the neuromodulation (based on baselined rewards for this trial) and apply plasticity
                    if numstep == STARTREWARDTIME:
                        # The network's response for this trial (0 or 1) is the index of the output neuron that had the highest cumulative output over the response period
                        responses = torch.argmax(totalresps, dim=1)  # responses is a 1D, integer-valued array of size BS. totalresps is a 2D real-vlued array of size BS, NBRESPS+1                           
                        
                        # blosses (binary losses) is a 1/-1 "correct/wrong" signal for each batch element for this trial.
                        blosses = 2.0 * (responses == torch.argmax(targets[:, :, 0], dim=1)).float() - 1.0    
                        responses0thisgen.append(float(responses[0]))

                        # We also want the 1-hot version of the response for each neuron. This will be used as the response signal below.
                        if numtrial > 0:
                            responses1hot_prev = responses1hot.clone()
                        responses1hot = F.one_hot(responses, 2)

                        # Now we apply lifetime plasticity, with node-perturbation, based on eligibility trace and suitably baselined reward/loss


                        # Baseline computation - only used for external neuromodulation experiments
                        # We compute separate baseline (running average) losses for different types of trials, as defined by their inputs (as in Miconi, eLife 2017). 
                        # So we need to find out the trial type for each element in batch.
                        # input1 = inputs[:, 0, 0]; input2 = inputs[:, 1, 0]  # Uh, what was that?
                        input1 = stims1[:,  0]; input2 = stims2[:, 0]  
                        trialtypes = (input1>0).long() * 2 + (input2>0).long()

                        if MODULTYPE == 'EXTERNAL' and numtrial > 30:                        
                            dw =  - (ETA * eligtraces  * (  meanlosstrace[np.arange(BS), trialtypes] * (mselossesthistrial - meanlosstrace[np.arange(BS), trialtypes]) )[:, None, None]).clamp(-MAXDW, MAXDW)
                            bpw += dw



                        # Updating the baseline - running average of losses, for each batch element, for the trial type just seen
                        meanlosstrace[torch.arange(BS).long(), trialtypes] *= MULOSSTRACE
                        meanlosstrace[torch.arange(BS).long(), trialtypes] +=  (1.0 - MULOSSTRACE) * mselossesthistrial




                    # Plasticity computation for internal (network-controlled) neuromodulation.
                    # Note that it is applied at every time step, unlike external neuromodulation experiments which only apply plasticity once per  trial, at the beginning of the reward period (see above).
                    if numtrial > 10 and MODULTYPE == 'INTERNAL':  # Lifetime plasticity is only applied after a few burn-in trials.
                        # eligtraces: BS x N x N (1 per connection & batch element)  mselossesthhistrial:  BS.    meanlosstrace: BS x (N.N).    trialtypes: BS    bresps/bstates:  BS x N 
                        # dw should have shape BS x N x N, i.e. one for each connection and batch element. Do not sum over batch dimension! The batch is purely evolutionary !

                        # Compute and apply the plasticity, based on accumulated eligibility traces and output of a certain neuron
                        if  numstep > 0:
                            modulsprev = moduls.clone()
                        moduls = bresps[:, MODNEURONS[0]] - bresps[:, MODNEURONS[1]]
                        # lifeactpens += torch.abs(moduls)
                        if numstep > 0 :
                            lifeactpens += (modulsprev - moduls) ** 2


                        # If we use only the first neuromodulatory neuron's (tanh) output as the actual neuromodulatory output:
                        # dw =   (ETA * eligtraces * bresps[:, MODNEURONS[0]][:, None, None] ).clamp(-MAXDW, MAXDW)

                        dw =   (ETA * eligtraces * moduls[:, None, None] ).clamp(-MAXDW, MAXDW)


                        bpw += dw



                    # Are we in the reward signal period?
                    # Note: the actual neuromodulatory reward signal (which influences plasticity) is applied just once per trial, above. Here we provide a binary "correct/ incorrect" signal to the network, 
                    # i.e. "was my response right or wrong for this trial?" 
                    # We also provide a signal indicating which response it gave in this trial (in theory it should be able to calculate it itself if needed, but this may help).
                    if numstep >= STARTREWARDTIME and numstep < ENDREWARDTIME: # Note that by this time, the loss has been computed and is fixed
                        
                        # # We provide a binary, "correct/incorrect" signal to the network
                        # bresps[:,REWARDNEURONS[0]] = REWARDSIZE * blosses[:]         # Reward input is also clamping
                        # bresps[:,REWARDNEURONS[1]] = -REWARDSIZE * blosses[:]         # Reward input is also clamping

                        # Akshully, we provide the same MSE loss that is used to guide evolution
                        # bresps[:,REWARDNEURONS[0]] = REWARDSIZE * mselossesthistrial[:]         # Reward input is also clamping
                        # bresps[:,REWARDNEURONS[1]] = -REWARDSIZE * mselossesthistrial[:]         # Reward input is also clamping

                        # Akshully^2, we duplicate the reward signal across many neurons to (maybe) increase its potnetial impact and exploitability (?...)
                        bresps[:,REWARDNEURONS[0]:REWARDNEURONS[-1]+1] = REWARDSIZE * mselossesthistrial[:, None]         # Reward input is also clamping
                        bresps[:,REWARDNEURONS[0]:REWARDNEURONS[-1]+1].clip_(min=0)            # Not sure if this helps. Well, obviously not if using plain MSE which is always +ve.
                        # bresps[:,REWARDNEURONS] = REWARDSIZE * mselossesthistrial[:, None]         # Reward input is also clamping
                        # bresps[:,REWARDNEURONS].clip_(min=0)            # Not sure if this helps. Well, obviously not if using plain MSE which is always +ve.




                        # We provide the network with a signal indicating the actual response it chose for this trial. Not sure if needed.  
                        # bresps[:, RESPSIGNALNEURONS] = responses1hot.float() * RESPSIGNALSIZE
                        bresps[:, RESPSIGNALNEURONS[0]:RESPSIGNALNEURONS[-1]+1] = responses1hot.float() * RESPSIGNALSIZE


                        
                    else:
                        bresps[:,REWARDNEURONS[0]:REWARDNEURONS[-1]+1] = 0
                        bresps[:, RESPSIGNALNEURONS[0]:RESPSIGNALNEURONS[-1]+1] = 0
                        # bresps[:,REWARDNEURONS] = 0
                        # bresps[:, RESPSIGNALNEURONS] = 0


                    
                    # modouts.append(float(moduls[0]))
                    # rewins.append(float(bresps[0, REWARDNEURONS[0]]))
                    if COLLECTMODOUTSANDREWINS:
                        stimz.append(bresps[0, STIMNEURONS[0]])
                        respz.append(bresps[0, RESPNEURONS[1]] - bresps[0, RESPNEURONS[0]])
                        if MODULTYPE == 'INTERNAL': #  Doesn't make sense  for external modulation
                            modouts.append(moduls[0])   
                        rewins.append(bresps[0, REWARDNEURONS[0]])
                    

                    if EVALW: 
                        allresps.append(bresps.cpu().numpy().astype('float32'))
                    # if EVALW and numtrial >= NBTRIALS - 50:
                    #     stimz.append(bresps[:, STIMNEURONS[0]])
                    #     respz.append(bresps[:, RESPNEURONS[1]] - bresps[:, RESPNEURONS[0]])
                    #     if MODULTYPE == 'INTERNAL': #  Doesn't make sense  for external modulation
                    #         modouts.append(moduls[:])
                    #     rewins.append(bresps[:, REWARDNEURONS[0]])



                # Now all steps done for this trial:
            
                if PRINTING:
                    if np.random.rand() < .1: 
                        print("|", int(responses[0]), int(blosses[0]), end=' ')
                
                ml0s.append(float(mselossesthistrial[0]))
                bl0s.append(float(blosses[0]))
                bls.append(blosses.cpu().numpy())


                # If this trial is part of the last NBTRIALSLOSS, we accumulate its trial loss into the agent's total loss for this task.
                if numtrial >= NBTRIALS - NBTRIALSLOSS:     # Lifetime  losses are only estimated over the last NBTRIALSLOSS trials
                    # taskmselosses += 2 * mselossesthistrial / NBTRIALSLOSS   # the 2* doesn't mean anything
                    taskmselosses +=  mselossesthistrial / NBTRIALSLOSS 
                    taskblosses += blosses / NBTRIALSLOSS
                    

            # Now all trials done for this task:
            if PRINTING:
                # print("Med task mseloss:", "{:.4f}".format(float(torch.median(taskmselosses))))
                print("\nTASK BLOSS[0]:", "{:.4f}".format(float(taskblosses[0])), "Med task bloss:", "{:.4f}".format(float(torch.median(taskblosses))), 
                "Med-abs totaldw[0]:", "{:.4f}".format(float(torch.median(torch.abs(bpw[0,:,:])))),
                "Max-abs totaldw[0]:", "{:.4f}".format(float(torch.max(torch.abs(bpw[0,:,:]))))
                )
            
            
            
            # raise ValueError

            if EVALW:
                pass
                # print("Saving Resps, Stims,  RI, MO")

                # np.savetxt('stims.txt', np.vstack([x.cpu().numpy() for x in stimz]))
                # np.savetxt('resps.txt', np.vstack([x.cpu().numpy() for x in respz]))
                # np.savetxt('modouts.txt', np.vstack([x.cpu().numpy() for x in modouts]))
                # np.savetxt('rewins.txt', np.vstack([x.cpu().numpy() for x in rewins]))


            if COLLECTMODOUTSANDREWINS:
                print("Saving Resps, Stims,  RI, MO")

                np.savetxt('stims.txt', np.array([float(x) for x in stimz]))
                np.savetxt('resps.txt', np.array([float(x) for x in respz]))
                np.savetxt('modouts.txt', np.array([float(x) for x in modouts]))
                np.savetxt('rewins.txt', np.array([float(x) for x in rewins]))

                # print("")
            lifemselosses += taskmselosses / NBTASKSPERGEN 
            lifeblosses += taskblosses / NBTASKSPERGEN 
        
            if (TESTING or numgen == 0) and numtask == 0:
                # These files contain respectively the first and *latest* Testing block of the *current* run only. 
                FNAME = 'bl_1standLastBlock_gen0.txt' if numgen == 0 else 'bl_1standLastBlock_lastgen.txt'
                # np.savetxt(FNAME, np.array(bl0s))
                np.savetxt(FNAME, np.vstack(bls))



        # After all tasks done for this lifetime / generation:

        lifeactpens /= (NBTASKSPERGEN * NBTRIALS)
        # lifeactpens -= torch.mean(lifeactpens); lifeactpens /= torch.std(lifeactpens)
        # lifeactpens += torch.mean(lifemselosses); lifeactpens *= torch.std(lifemselosses)
        
        lifelosses = lifemselosses + ALPHAACTPEN * lifeactpens

        binarylosses.append(float(lifeblosses[0]))
        evolosses.append(float(lifemselosses[0]))

        np.savetxt('blosses_onerun.txt', np.array(binarylosses))
        np.savetxt('mselosses_onerun.txt', np.array(evolosses))


        if EVALW:
            # Note: we use .npy format, because multi-dimensional.
            
            np.save('allstims.npy',  np.stack(allstims, -1))
            np.save('alltgts.npy',  np.stack(alltgts, -1))

            # print(len(allresps), len(allstims), len(alltgts))
            assert len(allresps) == NBTRIALS * T
            # print(allresps[0].shape, allstims[0].shape, alltgts[0].shape)
            print("Rearranging saved responses into appropriate shape...")
            z1 = np.dstack(allresps)
            z2 = np.stack(np.split(z1, NBTRIALS, axis=2), axis=-1)
            print("Final shape of the saved responses:", z2.shape)
            assert(z2.shape == (BS, N, T, NBTRIALS))
            np.save('allresps.npy', z2[:,:,:,[9,-1]]) # We only store response data for 9th (before plasticity starts) and last trial (to keep file size manageable) 



        # Now we're ready to perform evolution (by computing gradients by hand, and then applying the optimizer with these gradients)
        optimizer.zero_grad()

        # Gradient is just loss x mutation (remember we use antithetic sampling)
        # gradient = torch.sum(mutations_wandalpha[0] * lifelosses[:, None, None], axis=0) # / BS
        gradient = torch.sum(mutations_wandalpha[0] * lifelosses[:, None, None], axis=0)  / (BS * MUTATIONSIZE * MUTATIONSIZE)

        
        # gradient = gradient / 100


        wgradnorm = float(torch.norm(gradient))
        wgradnorms.append(wgradnorm)
        if PRINTING:
            print("norm w:", "{:.4f}".format(float(torch.norm(w))), "norm gradient:", "{:.4f}".format(wgradnorm), 
                  "med-abs w:", "{:.4f}".format(float(torch.median(torch.abs(w)))), 
                  "max-abs w:", "{:.4f}".format(float(torch.max(torch.abs(w)))), 
                    "norm a:", "{:.4f}".format(float(torch.norm(alpha))), "mean a:",  "{:.4f}".format(float(torch.mean(alpha))))


        w.grad = gradient
        wprev = w.clone()

        # gradientalpha = torch.sum(mutations_wandalpha[1] * lifelosses[:, None, None], axis=0) # / BS
        gradientalpha = torch.sum(mutations_wandalpha[1] * lifelosses[:, None, None], axis=0)  / (BS * MUTATIONSIZE * MUTATIONSIZE)


        # gradientalpha = gradientalpha / 100


        alpha.grad = gradientalpha
        alphaprev = alpha.clone()

        if numgen > 0 and not TESTING and not EVALW:
            optimizer.step()

        
        wdiff = w - wprev
        adiff = alpha - alphaprev
        if PRINTING:
            print("Norm w-wprev:", "{:.4f}".format(float(torch.norm(wdiff))), "Max abs w-wprev:", "{:.4f}".format(float(torch.max(torch.abs(wdiff)))), 
                "Norm a-aprev:", "{:.4f}".format(float(torch.norm(adiff))), "Max abs a-aprev:", "{:.4f}".format(float(torch.max(torch.abs(adiff))))  )

    

        if PRINTING:
            print("Med/min/max/Half-Nth/0th loss in batch:", float(torch.median(lifelosses)), float(torch.min(lifelosses)), float(torch.max(lifelosses)),
                                    float(lifelosses[BS//2]), float(lifelosses[0]))
            print("Med/min/max/Half-Nth/0th life mse loss in batch:", float(torch.median(lifemselosses)), float(torch.min(lifemselosses)), float(torch.max(lifemselosses)),
                                    float(lifemselosses[BS//2]), float(lifemselosses[0]))
            print("Med/min/max/Half-Nth/0th activity penalty in batch:", float(torch.median(lifeactpens)), float(torch.min(lifeactpens)), float(torch.max(lifeactpens)),
                                    float(lifeactpens[BS//2]), float(lifeactpens[0]))
            print("Gen", numgen, "done in", time.time()-tic)
    





print("Time taken:", time.time()-ticstart)
