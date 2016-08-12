--------------------------------------------------------------------------------
-- plaindataset: wraps X_train, Y_train, X_test, Y_test
--               to give a unified interface.
--------------------------------------------------------------------------------

local PlainDataset = torch.class 'PlainDataset'

function PlainDataset:__init(args)
    -- parse args
    local argnames = {'X_train','Y_train','X_test','Y_test'}
    for _,v in pairs(argnames) do
        self[v] = args[v]
    end

    self.ntrain = self.X_train:size(1)
    self.ntest = self.X_test:size(1)
end

---------------------------------------------------------------
-- load training batch sample
--
function PlainDataset:sample(quantity)
    local indices = torch.LongTensor(quantity):random(self.ntrain)
    local X = self.X_train:index(1, indices)
    local Y = self.Y_train:index(1, indices)
    return X, Y
end

---------------------------------------------------------------
-- load test batch sample
--
function PlainDataset:get(i1,i2)
    local indices = torch.range(i1,i2):long()
    local X = self.X_test:index(1, indices)
    local Y = self.Y_test:index(1, indices)
    return X, Y
end
