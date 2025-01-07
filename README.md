# Objective
The goal of this project is to estimate the **maximum speed of a volleyball** using video footage.

# Method
Using the known **volleyball diameter** (~21 cm), its pixel size in the video, and camera properties such as **focal length** and **sensor size**, we can estimate the ball's distance from the camera. By analyzing the change in this distance over time, we can calculate the volleyball's velocity in the direction of the camera.

Given the availability of high-fps and high-quality video from modern cameras, this approach enables precise speed estimation.

The model is composed of two main components:
1. **Volleyball Detector** - A YOLO-based model designed to detect the volleyball in the video
2. **Velocity Estimator** - A simple machine learning algorithm that calculates the maximum speed of the volleyball.

# Evaluation set
The evaluation set includes:
**9 videos** with varied characteristics:
- Distinct backgrounds
- Different ball-throwing movements
- Wide range of ball speed

Each video includes **measured volleyball speed** obtained from a radar gun, providing a ground truth for comparison.

# Human benchmark
Before implementing AI-based methods, it's crucial to establish whether human experts can perform this task accurately. If humans cannot achieve reasonable performance, it is unlikely that an AI model will perform well either (aligned with Google's philosophy that AI should automate tasks already achievable by humans).

For the human benchmark:
1. The ball was manually labeled in the video frames.
2. Distances were computed based on these labels.
3. The appropriate video fragments were analyzed to estimate maximum speed.
4. These estimates were compared with radar gun measurements.

Results:
- **Mean Absolute Percentage Error (MAPE)**: 6.8%
- **Root Mean Square Error (RMSE)**: 4.2 km/h

Assuming a normal error distribution, approximately **95% of the predictions** are expected to have an absolute error of less than **8.4 km/h**.

# Model performance
The model was evaluated using the same dataset, yielding the following results:

- **Mean Absolute Percentage Error (MAPE)**: 7.6%
- **Root Mean Square Error (RMSE)**: 4.6 km/h

# Conclusions
Although the evaluation dataset is relatively small, limiting the conclusiveness of the results, the model shows strong potential by approaching **human-level performance** and appears promising for practical applications.

# Future Work
Additional details and updates will be added soon. Stay tuned!
