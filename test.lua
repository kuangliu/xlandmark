------------------------------------------------------------------------
-- test example with iterative refinement process.
------------------------------------------------------------------------

require 'nn';
require 'xlua';
require 'image';

function plotim(points, im, savename)
    H,W,_ = unpack(im:size():totable())
    for i = 1,5 do
        local x = math.ceil(points[2*i-1])
        local y = math.ceil(points[2*i])
        -- print(x,y)
        im = image.drawRect(im, x, y, x+2, y+2, {lineWidth=2})
    end
    image.save('./rlt/'..savename, im)
end

function forward(net, im)
    -- rescale to 40x40
    local H = im:size(2)
    local W = im:size(3)
    local im96 = image.scale(im,96,96)

    -- zero mean
    local x = im96:add(-trainMean):cdiv(trainStd)
    -- RGB -> BGR
    --x = x:index(1,torch.LongTensor{3,2,1})

    -- forward network
    local y = net:forward(x)
    y:add(1):mul(48)
    for i = 1,5 do
        y[2*i-1] = math.ceil(y[2*i-1]*W/96)
        y[2*i] = math.ceil(y[2*i]*H/96)
    end
    return y
end

function get2x(pts, im)
    local xs = pts:index(1,torch.LongTensor({1,3,5,7,9}))
    local ys = pts:index(1,torch.LongTensor({2,4,6,8,10}))

    local H = im:size(2)
    local W = im:size(3)

    local xmax = xs:max()
    local xmin = xs:min()

    local ymax = ys:max()
    local ymin = ys:min()

    local Wbox = xmax-xmin
    local Hbox = ymax-ymin

    local xmin2 = math.max(1, math.ceil(xmin-Wbox/2))
    local xmax2 = math.min(W, math.ceil(xmax+Wbox/2))
    local ymin2 = math.max(1, math.ceil(ymin-Hbox/2))
    local ymax2 = math.min(H, math.ceil(ymax+Hbox/2))

    return xmin2,xmax2,ymin2,ymax2
end


net = torch.load('./model/bestnn.t7')

trainMean = torch.load('./model/trainMean.t7')
trainStd = torch.load('./model/trainStd.t7')
trainMean = image.scale(trainMean, 96, 96)
trainStd = image.scale(trainStd, 96, 96)

imPath = '/mnt/hgfs/D/dataset/818/all/'

for imName in paths.iterfiles(imPath) do
    print(imName)
    im = image.load(imPath..imName)

    H = im:size(2)
    W = im:size(3)
    y = forward(net, im)

    xmin = 0
    ymin = 0
    for i = 1,3 do -- iterate for 3 times
        for i = 1,5 do
            y[2*i-1] = y[2*i-1] + xmin
            y[2*i] = y[2*i] + ymin
        end

        -- get a 2x region
        xmin2,xmax2,ymin2,ymax2 = get2x(y, im)
        xmin = xmin2
        ymin = ymin2
        im2 = im[{{},{ymin2,ymax2},{xmin2,xmax2}}]

        -- forward again
        y = forward(net, im2)
        plotim(y, im2, imName..'_'..tostring(i)..'.jpg')
    end
    break
end
