# My3DGNN
My3DGNN【个人版】
# train
- batchsize = 2
- eopoch = 5

```python
x = rgbd_label_xy[0] # [2, 640, 480, 6]
target = rgbd_label_xy[1].long() # [2, 640, 480]
xy = rgbd_label_xy[2] # [2, 640, 480, 2]

input = x.permute(0, 3, 1, 2) # [2, 6, 640, 480]
xy = xy.permute(0, 3, 1, 2).contiguous() # [2, 2, 640, 480]

output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy, use_gnn=config.use_gnn)
output # [2, 14, 640, 480]
```