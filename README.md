![Screenshot (2)](https://user-images.githubusercontent.com/66817101/217295454-7146e440-7027-4b93-87cd-b7ce8897fb60.png)
# Unsupervised-Learning-Recommenders-Reinforcement-Learning
We are now ready to train our agent to solve the Lunar Lander environment. In the cell below we will implement the algorithm in Fig 3 line by line (please note that we have included the same algorithm below for easy reference. This will prevent you from scrolling up and down the notebook):

Line 1: We initialize the memory_buffer with a capacity of  𝑁=
  MEMORY_SIZE. Notice that we are using a deque as the data structure for our memory_buffer.
Line 2: We skip this line since we already initialized the q_network in Exercise 1.
Line 3: We initialize the target_q_network by setting its weights to be equal to those of the q_network.
Line 4: We start the outer loop. Notice that we have set  𝑀=
  num_episodes = 2000. This number is reasonable because the agent should be able to solve the Lunar Lander environment in less than 2000 episodes using this notebook's default parameters.
Line 5: We use the .reset() method to reset the environment to the initial state and get the initial state.
Line 6: We start the inner loop. Notice that we have set  𝑇=
  max_num_timesteps = 1000. This means that the episode will automatically terminate if the episode hasn't terminated after 1000 time steps.
Line 7: The agent observes the current state and chooses an action using an  𝜖
 -greedy policy. Our agent starts out using a value of  𝜖=
  epsilon = 1 which yields an  𝜖
 -greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed state. As training progresses we will decrease the value of  𝜖
  slowly towards a minimum value using a given  𝜖
 -decay rate. We want this minimum value to be close to zero because a value of  𝜖=0
  will yield an  𝜖
 -greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the action that it believes (based on its past experiences) will maximize  𝑄(𝑠,𝑎)
 . We will set the minimum  𝜖
  value to be 0.01 and not exactly 0 because we always want to keep a little bit of exploration during training. If you want to know how this is implemented in code we encourage you to take a look at the utils.get_action function in the utils module.
Line 8: We use the .step() method to take the given action in the environment and get the reward and the next_state.
Line 9: We store the experience(state, action, reward, next_state, done) tuple in our memory_buffer. Notice that we also store the done variable so that we can keep track of when an episode terminates. This allowed us to set the  𝑦
  targets in Exercise 2.
Line 10: We check if the conditions are met to perform a learning update. We do this by using our custom utils.check_update_conditions function. This function checks if  𝐶=
  NUM_STEPS_FOR_UPDATE = 4 time steps have occured and if our memory_buffer has enough experience tuples to fill a mini-batch. For example, if the mini-batch size is 64, then our memory_buffer should have more than 64 experience tuples in order to pass the latter condition. If the conditions are met, then the utils.check_update_conditions function will return a value of True, otherwise it will return a value of False.
Lines 11 - 14: If the update variable is True then we perform a learning update. The learning update consists of sampling a random mini-batch of experience tuples from our memory_buffer, setting the  𝑦
  targets, performing gradient descent, and updating the weights of the networks. We will use the agent_learn function we defined in Section 8 to perform the latter 3.
Line 15: At the end of each iteration of the inner loop we set next_state as our new state so that the loop can start again from this new state. In addition, we check if the episode has reached a terminal state (i.e we check if done = True). If a terminal state has been reached, then we break out of the inner loop.
Line 16: At the end of each iteration of the outer loop we update the value of  𝜖
 , and check if the environment has been solved. We consider that the environment has been solved if the agent receives an average of 200 points in the last 100 episodes. If the environment has not been solved we continue the outer loop and start a new episode.
Finally, we wanted to note that we have included some extra variables to keep track of the total number of points the agent received in each episode. This will help us determine if the agent has solved the environment and it will also allow us to see how our agent performed during training. We also use the time module to measure how long the training takes.




Fig 4. Deep Q-Learning with Experience Replay.

Note: With this notebook's default parameters, the following cell takes between 10 to 15 minutes to run.
