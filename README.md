# xlandmark: facial landmark regression with xtorch
`xlandmark` is a facial landmark regression model with end-to-end training.  

## dataset
`xlandmark` is trained on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset, with 5 landmark (10 coordinates) outputs.  

TODO: dataset process

## model & training
### model
The original model is from [VanillaCNN](https://github.com/ishay2b/VanillaCNN), with 2 modifications:
1. Input size changed from `40x40` to `96x96`.
2. FC layer output changed accordingly.

Modified model architecture:
```lua
nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): nn.Copy
  (2): DataParallelTable: 4 x nn.Sequential {
    [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> output]
    (1): cudnn.SpatialConvolution(3 -> 16, 5x5, 1,1, 2,2) without bias
    (2): cudnn.Tanh
    (3): nn.Abs
    (4): cudnn.SpatialMaxPooling(2x2, 2,2)
    (5): cudnn.SpatialConvolution(16 -> 48, 3x3, 1,1, 1,1) without bias
    (6): cudnn.Tanh
    (7): nn.Abs
    (8): cudnn.SpatialMaxPooling(2x2, 2,2)
    (9): cudnn.SpatialConvolution(48 -> 64, 3x3) without bias
    (10): cudnn.Tanh
    (11): nn.Abs
    (12): cudnn.SpatialMaxPooling(3x3, 2,2)
    (13): cudnn.SpatialConvolution(64 -> 64, 2x2) without bias
    (14): cudnn.Tanh
    (15): nn.Abs
    (16): nn.View(-1)
    (17): nn.Linear(6400 -> 1024)
    (18): cudnn.Tanh
    (19): nn.Abs
    (20): nn.Linear(1024 -> 10)
  }
}

```

### training
[`xtorch`](https://github.com/kuangliu/xtorch) is used for data loading and training.  
- `MSECriterion` is used instead of `CrossEntropyCriterion`.
- `ConfusionMatrix` is removed for regression task.
- `testLoss` is used as the standard to save best checkpoint.

## iterative refinement
TODO
