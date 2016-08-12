require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'pl'
require 'paths'

utils = dofile('utils.lua')
xtorch = dofile('xtorch.lua')

------------------------------------------------
-- 1. prepare data
--

dofile('listdataset.lua')
ds = ListDataset({
    trainData = '/search/ssd/liukuang/celeba/train/',
    trainList = '/search/ssd/liukuang/celeba/train.txt',
    testData = '/search/ssd/liukuang/celeba/test/',
    testList = '/search/ssd/liukuang/celeba/test.txt',
    imsize = 96
})
------------------------------------------------
-- 2. define net
--
net = torch.load('./model.t7')

------------------------------------------------
-- 3. init optimization params
--
optimState = {
    learningRate = 0.001,
    learningRateDecay = 1e-7,
    weightDecay = 1e-4,
    momentum = 0.9,
    nesterov = true,
    dampening = 0.0
}

opt = {
    ----------- net options --------------------
    net = net,
    ----------- data options -------------------
    dataset = ds,
    nhorse = 4,   -- nb of threads to load data, default 1
    ----------- training options ---------------
    batchSize = 128,
    nEpoch = 500,
    ----------- optimization options -----------
    optimizer = optim.adam,
    criterion = nn.MSECriterion,
    optimState = optimState,
    ----------- general options ----------------
    backend = 'GPU',    -- CPU or GPU, default CPU
    nGPU = 4,
    verbose = true
}

------------------------------------------------
-- 4. and fit!
--
xtorch.fit(opt)
