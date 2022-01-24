# Car Racing World Model

A world model for the car racing envrionment in OpenAI Gym 

A generative model based on Order Agnostic Diffusion Models

To create a dataset

Drive the car and record an episode
```bash
python drive.py -ep 0
```

Compute locally consistent poses for the episode using icp algorithm
```bash
python generate_pose.py -ep 0
```

Fuse the trajectory into egocentric occupancy maps for each timestep
```bash
python rolling_window.py -ep 0
```

Generate a segment map and a signed distance function for each timestep
```bash
python seg_to_sdf.py -ep 0
```