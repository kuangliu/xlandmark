require 'os';
require 'sys';
require 'xlua';
require 'image';
require 'torch';
require 'paths';
ffi = require 'ffi';

torch.setdefaulttensortype('torch.FloatTensor')

local ListDataLoader = torch.class 'ListDataLoader'
local pathcat = paths.concat

---------------------------------------------------------------------------
-- DataLoader takes:
--  - dataPath: a folder containing the images.
--  - listPath: a text file containing sample names & targets.
--  - imfunc: the image processing function
--
function ListDataLoader:__init(dataPath, listPath, imfunc)
    assert(paths.dirp(dataPath), dataPath..' not exist!')
    assert(paths.filep(listPath), listPath..' not exist!')
    self.dataPath = dataPath
    self.listPath = listPath
    -- set the default image processing function, just resize to loadsize
    self.__imfunc = imfunc

    self:__parseList()
end

---------------------------------------------------------------------------
-- parse list files, split file names and targets.
-- returns:
--    - names: 2D tensor containing the names sized [N, maxNameLength]
--    - targets: 2D tensor containing the targets sized [N, D]
-- where
--   - N: # of samples
--   - maxNameLength: max length of all names
--   - D: dims of the target
--
-- We first pre-allocate names sized [N, constLength],
-- and trim to [N, maxNameLength] later.
--
function ListDataLoader:__parseList()
    print('parsing list...')

    local constLength = 50           -- assume the length of all file names < constLength
    local maxNameLength = -1         -- max file name length

    -- get the number of files
    local N = tonumber(sys.fexecute('ls '..self.dataPath..' | wc -l'))
    self.nSamples = N

    local names = torch.CharTensor(N,constLength):fill(0)
    local targets

    -- parse names and targets line by line
    local name_data = names:data()
    local f = assert(io.open(self.listPath, 'r'))
    for i = 1,N do
        xlua.progress(i,N)
        local line = f:read('*l')

        local splited = string.split(line, '%s+')
        ffi.copy(name_data, splited[1])    -- image name
        name_data = name_data + constLength
        maxNameLength = math.max(maxNameLength, #splited[1])

        local target = {}    -- targets
        for i = 2,#splited do
            target[#target+1] = tonumber(splited[i])
        end

        targets = targets or torch.Tensor(N, #target)
        targets[i] = torch.Tensor(target)
    end
    f:close()

    -- trim names from [N,constLength] -> [N,maxNameLength]
    names = names[{ {},{1,maxNameLength} }]
    self.names = names
    self.targets = targets
end

--------------------------------------
-- load images from the given indices.
--
function ListDataLoader:__loadImages(indices)
    local quantity = indices:nElement()
    local images
    for i = 1,quantity do
        local name = ffi.string(self.names[indices[i]]:data())
        local im = image.load(pathcat(self.dataPath, name))
        -- hooker function performing like resizing, zero-mean, normalization
        im = self.__imfunc(im)
        images = images or torch.Tensor(quantity, 3, im:size(2), im:size(2))
        images[i] = im
    end
    return images
end

---------------------------------------------------------
-- randomly sample quantity images from training dataset.
-- load samples maybe overlapped.
--
function ListDataLoader:sample(quantity)
    assert(quantity, '[ERROR] => No sample quantity specified!')
    local indices = torch.LongTensor(quantity):random(self.nSamples)
    local images  = self:__loadImages(indices)
    local targets = self.targets:index(1,indices)
    return images, targets
end

------------------------------------------------------------
-- return a batch specified by indices.
--
function ListDataLoader:loadBatchByIndex(indices)
    local images = self:__loadImages(indices)
    local targets = self.targets:index(1,indices)
    return images, targets
end

------------------------------------------------
-- get images in the index range [i1, i2]
-- used to load test samples.
--
function ListDataLoader:get(i1,i2)
    local indices = torch.range(i1,i2):long()
    local images = self:__loadImages(indices)
    local targets = self.targets:index(1,indices)
    return images, targets
end
