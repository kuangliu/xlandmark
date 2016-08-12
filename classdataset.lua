--------------------------------------------------------------------------------
-- classdataset loads training/test data from disk. But unlike listdataset,
-- there is no index list, the images are organized in subfolders, the subfolder
-- name is the class name.
--
-- Directory arrangement:
-- -- train
--      |_ class 1
--      |_ class 2
--      |_ ...
-- -- test
--      |_ class 1
--      |_ class 2
--      |_ ...
--------------------------------------------------------------------------------
dofile('./dataloader/classdataloader.lua')

local ClassDataset = torch.class 'ClassDataset'

function ClassDataset:__init(args)
    -- parse args
    local argnames = {'trainData', 'testData', 'imsize'}
    for _,v in pairs(argnames) do
        self[v] = args[v]
    end

    -- init trainloader & testLoder
    self:__initLoader('trainLoader', self.trainData)
    self:__initLoader('testLoader', self.testData)
    self:__calcMeanStd() -- compute trainLoader mean & std

    self.ntrain = self.trainLoader.nSamples
    self.ntest = self.testLoader.nSamples

    -- update imfunc, perform zero mean & normalization
    local imfunc = function(im)
        -- resize
        im = image.scale(im, self.imsize, self.imsize)
        -- zero-mean & normalization
        for i = 1,3 do  -- for RGB channel
            im[i]:add(-self.mean[i])
            im[i]:div(self.std[i])
        end
        return im
    end

    self.trainLoader.__imfunc = imfunc
    self.testLoader.__imfunc = imfunc

    -- save
    if not paths.filep('./cache/trainLoader.t7') then
        torch.save('./cache/trainLoader.t7', self.trainLoader)
    end

    if not paths.filep('./cache/testLoader.t7') then
        torch.save('./cache/testLoader.t7', self.testLoader)
    end
end

---------------------------------------------------------------
-- init trainLoader & testLoader
--
function ClassDataset:__initLoader(loaderName, dataPath)
    paths.mkdir('cache')

    -- set the default image processing function to resize
    local imscale = function(im) return image.scale(im, self.imsize, self.imsize) end

    local filePath = './cache/'..loaderName..'.t7'
    if paths.filep(filePath) then
        print('==> loading '..loaderName..' from cache...')
        loader = torch.load(filePath)
    else
        print('==> init '..loaderName..'...')
        loader = ClassDataLoader(dataPath, imscale)
    end
    self[loaderName] = loader
end

---------------------------------------------------------------
-- calculate training mean & std
--
function ClassDataset:__calcMeanStd()
    local filePath = './cache/meanstd.t7'
    local cache, mean, std
    if paths.filep(filePath) then
        print('==> loading mean & std from cache...')
        cache = torch.load(filePath)
        mean = cache.mean
        std = cache.std
    else
        print('==> computing mean & std...')
        local nSamples = math.min(10000, self.trainLoader.nSamples)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for i = 1,nSamples do
            xlua.progress(i,nSamples)
            local im = self.trainLoader:sample(1)[1]
            for j = 1,3 do
                mean[j] = mean[j] + im[j]:mean()
                std[j] = std[j] + im[j]:std()
            end
        end
        cache = {}
        cache.mean = mean:div(nSamples)
        cache.std = std:div(nSamples)
        torch.save(filePath, cache)
    end
    self.mean = mean
    self.std = std
end

---------------------------------------------------------------
-- load training batch sample
--
function ClassDataset:sample(quantity)
    return self.trainLoader:sample(quantity)
end

---------------------------------------------------------------
-- load test batch sample
--
function ClassDataset:get(i1,i2)
    return self.testLoader:get(i1,i2)
end
