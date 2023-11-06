# NHL Neural Network

### Currently (last updated June 2023):
1. Parse and process game data from NHL API from 2011 - 2022
   - Calculate number of seconds into the period
   - Calculate angle and distance to the net from location on ice
   - Process strings of event types (shot, goal, shot missed net, hit, takeway etc) for downstream text vectorization
   - Include previous events time, distance, angle, event type
   - Calculate linear speed from previous event to shot
   - Calculate angular speed from previous event to shot
   - Binary for empty net or not
2. Apply text vectorization on shot type and one-hot encode it 
3. Build tensor for input into NN
   - Shot type and previous event type with other numerical data from above
4. Build variety of densely connected DLNN models with Dropout
   - Variety of hidden layers and number of nodes in those layers
   - Either relu, elu, or tanh activation of dense layers
   - All sigmoid activation in output layer with 1 node (to predict shot percentage)
5. Compile and train models
   - ADAM optimizer while hyperbolically reducing learning rate as training progresses
     - Plot this learning schedule
   - Track loss, accuracy, binary cross entropy
   - Early stoppage to prevent overfitting (based on validation binary cross entropy with some patience)
   - Class weights applied (imbalance in shots that are goals and those that aren't)
6. Evaluate models
   - Most recent year is validation set
   - Plot loss for training set and validation set
7. Show random set of model predictions from validation set
