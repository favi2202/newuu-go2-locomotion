# Prediction before the wider-range experiment

I expect slightly worse speed tracking when the linear command ranges increase from [-1, 1] to [-2, 2] m/s with the same 300 training iterations. The robot may struggle to achieve the faster requested speeds, causing larger tracking errors and lower tracking rewards.

I think additional training may help it learn movements that follow the faster commands more accurately, but we need to test this.
