import torch
import torch.nn as nn
# include whatever other imports you need here

class WallyNetwork(nn.Module):
   # Your network definition goes here
   pass

def predict(images):
   # images: B x 3 x 256 x 256 tensor of RGB scenes with values in (0,1)

   # Determine which device the input tensor is on
   device = torch.device("cuda" if images.is_cuda else "cpu")

   model = WallyNetwork() # Add your model init parameters here if you have any
   # Move to same device as input
   model = model.to(device)
   # Load network weights
   model.load_state_dict(torch.load('q2_weights.pkl',map_location=torch.device(device)))
   # Put model in evaluation mode
   model.eval()

   # Optional: do whatever preprocessing you do on the inputs
   # if not included as transformations inside the model

   with torch.no_grad():
       # Pass inputs to model
       output = model(images)

   # Post-process the network output into detections:
   # H-level: B x 5 float tensor, rows (p, cx, cy, w, h)
   # M-level: B x 6 float tensor, rows (p, cx, cy, w, h, theta), theta in degrees
   # p is your score (higher = more confident my face is present). You must
   # return a box (and, at M-level, an angle) for every scene.
   detections = output

   return detections
