# NHL Neural Network

### Currently (last updated January 2024):
1. Parse and process shot data from play by play datasets from NHL API for all games 2011 - 2023
   - Calculate number of seconds into the period
   - Calculate angle and distance to the net from location on ice
   - Process strings for downstream text vectorization
     - event types (shot, goal, shot missed net, hit, takeway etc)
     - players involved (shooter and goalie)
     - on-ice situation (PP, even strength, goalie pulled etc)
     - type of shot (wrist, slap etc)
     - zone of event occurance (offensive, neutral, defensive)
     - team of shooter
   - Include previous event's time, distance, angle, event type, zone
   - Calculate time since previous event
   - Calculate linear and angular speed from previous event
2. Apply text vectorization on strings and one-hot encode them
3. Build tensor for input into NN
   - Combine one-hotted categorical variables with numerical variables
4. Build variety of densely connected DLNN models with Dropout
   - Variety of hidden layers and number of nodes in those layers
   - Either relu, elu, or tanh activation of dense layers
   - All sigmoid activation in output layer with 1 node (to predict shot percentage)
5. Compile and train models
   - ADAM optimizer while hyperbolically reducing learning rate as training progresses
     - Plot this learning schedule (current WIP)
   - Track loss, accuracy, binary cross entropy
   - Early stoppage to prevent overfitting (based on validation binary cross entropy with some patience)
   - Class weights applied (imbalance in shots that are goals and those that aren't)
6. Evaluate trained models
   - Most recent year is validation set
   - Plot loss for training set and validation set (current WIP)
7. Show random set of model predictions from validation set