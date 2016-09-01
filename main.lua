require 'pl'
require 'nn'
require 'nnx'
require 'xlua'
require 'torch'
require 'optim'
require 'image'
require 'paths'

utils = dofile('utils.lua')
xtorch = dofile('xtorch.lua')

------------------------------------------------
-- 1. prepare data
--
dofile('./datagen/datagen.lua')
dofile('./datagen/dataloader/listdataloader.lua')

trainloader = ListDataLoader({
    directory = '/search/ssd/liukuang/celeba/train/',
    list = '/search/ssd/liukuang/celeba/train.txt',
    imsize = 96
})

testloader = ListDataLoader({
    directory = '/search/ssd/liukuang/celeba/test/',
    list = '/search/ssd/liukuang/celeba/test.txt',
    imsize = 96
})

traindata = DataGen({
    dataloader=trainloader,
    standardize=true
})
mean,std = traindata:getmeanstd()

testdata = DataGen({
    dataloader=testloader,
    standardize={ mean=mean, std=std }
})

paths.mkdir('cache')
torch.save('./cache/traindata.t7',traindata)
torch.save('./cache/testdata.t7',testdata)
-- traindata = torch.load('./cache/traindata.t7')
-- testdata = torch.load('./cache/testdata.t7')

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
    traindata = traindata,
    testdata = testdata,
    nhorse = 8  -- nb of threads to load data, default 1
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
opt = xlua.envparams(opt)

------------------------------------------------
-- 4. and fit!
--
xtorch.fit(opt)
