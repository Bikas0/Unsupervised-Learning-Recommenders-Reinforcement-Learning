# Unsupervised-Learning-Recommenders-Reinforcement-Learning
We are now ready to train our agent to solve the Lunar Lander environment. In the cell below we will implement the algorithm in Fig 3 line by line (please note that we have included the same algorithm below for easy reference. This will prevent you from scrolling up and down the notebook):

Line 1: We initialize the memory_buffer with a capacity of  𝑁= MEMORY_SIZE. Notice that we are using a deque as the data structure for our memory_buffer.

Line 2: We skip this line since we already initialized the q_network in Exercise 1.

Line 3: We initialize the target_q_network by setting its weights to be equal to those of the q_network.

Line 4: We start the outer loop. Notice that we have set  𝑀= num_episodes = 2000. This number is reasonable because the agent should be able to solve the Lunar Lander environment in less than 2000 episodes using this notebook's default parameters.

Line 5: We use the .reset() method to reset the environment to the initial state and get the initial state.

Line 6: We start the inner loop. Notice that we have set  𝑇= max_num_timesteps = 1000. This means that the episode will automatically terminate if the episode hasn't terminated after 1000 time steps.

Line 7: The agent observes the current state and chooses an action using an  𝜖-greedy policy. Our agent starts out using a value of  𝜖= epsilon = 1 which yields an  𝜖
 -greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed state. As training progresses we will decrease the value of  𝜖 slowly towards a minimum value using a given  𝜖-decay rate. We want this minimum value to be close to zero because a value of  𝜖=0 will yield an  𝜖-greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the action that it believes (based on its past experiences) will maximize  𝑄(𝑠,𝑎). We will set the minimum  𝜖 value to be 0.01 and not exactly 0 because we always want to keep a little bit of exploration during training. If you want to know how this is implemented in code we encourage you to take a look at the utils.get_action function in the utils module.
 
 ![bike](https://user-images.githubusercontent.com/66817101/217494288-000be208-6c7c-476b-9848-31c2b62c790b.png)
 
Line 8: We use the .step() method to take the given action in the environment and get the reward and the next_state.

Line 9: We store the experience(state, action, reward, next_state, done) tuple in our memory_buffer. Notice that we also store the done variable so that we can keep track of when an episode terminates. This allowed us to set the  𝑦 targets in Exercise 2.

Line 10: We check if the conditions are met to perform a learning update. We do this by using our custom utils.check_update_conditions function. This function checks if  𝐶= NUM_STEPS_FOR_UPDATE = 4 time steps have occured and if our memory_buffer has enough experience tuples to fill a mini-batch. For example, if the mini-batch size is 64, then our memory_buffer should have more than 64 experience tuples in order to pass the latter condition. If the conditions are met, then the utils.check_update_conditions function will return a value of True, otherwise it will return a value of False.

Lines 11 - 14: If the update variable is True then we perform a learning update. The learning update consists of sampling a random mini-batch of experience tuples from our memory_buffer, setting the  𝑦 targets, performing gradient descent, and updating the weights of the networks. We will use the agent_learn function we defined in Section 8 to perform the latter 3.

Line 15: At the end of each iteration of the inner loop we set next_state as our new state so that the loop can start again from this new state. In addition, we check if the episode has reached a terminal state (i.e we check if done = True). If a terminal state has been reached, then we break out of the inner loop.

Line 16: At the end of each iteration of the outer loop we update the value of  𝜖, and check if the environment has been solved. We consider that the environment has been solved if the agent receives an average of 200 points in the last 100 episodes. If the environment has not been solved we continue the outer loop and start a new episode.

Finally, we wanted to note that we have included some extra variables to keep track of the total number of points the agent received in each episode. This will help us determine if the agent has solved the environment and it will also allow us to see how our agent performed during training. We also use the time module to measure how long the training takes.


![Screenshot (2)](https://user-images.githubusercontent.com/66817101/217295454-7146e440-7027-4b93-87cd-b7ce8897fb60.png)



# Unsupervised Learning

![Screenshot (49)](https://user-images.githubusercontent.com/66817101/224567385-d070f91d-fe40-4d46-a759-684f9e64d5b8.png)
![Screenshot (50)](https://user-images.githubusercontent.com/66817101/224567389-d8b988a1-9a71-475b-89dd-669c9748e7e9.png)
![Screenshot (51)](https://user-images.githubusercontent.com/66817101/224567391-6553d1b5-bed7-41c6-93e4-259826f467f7.png)
![Screenshot (52)](https://user-images.githubusercontent.com/66817101/224567452-d691033a-fd63-4ad6-9f3a-bb72bb7b132c.png)
![Screenshot (53)](https://user-images.githubusercontent.com/66817101/224567458-98e96d0d-f53a-40f2-ac02-85470bebad25.png)
![Screenshot (54)](https://user-images.githubusercontent.com/66817101/224567459-a0b45cf3-1885-4e6c-8fd9-efe1be023e17.png)
![Screenshot (55)](https://user-images.githubusercontent.com/66817101/224567460-2a1c209c-f77c-4880-8625-3afb9ea11a0e.png)
![Screenshot (56)](https://user-images.githubusercontent.com/66817101/224567463-8e9551a0-1cd3-45ff-a8b7-1fe6cf84e359.png)
![Screenshot (57)](https://user-images.githubusercontent.com/66817101/224567464-0a0706d1-8633-405c-a4e8-fa4b847ec3e0.png)
![Screenshot (58)](https://user-images.githubusercontent.com/66817101/224567467-bfd40432-ba2f-42d1-b1d7-f71d43d07910.png)
![Screenshot (59)](https://user-images.githubusercontent.com/66817101/224567469-6b60b726-e4e1-4bb9-a907-523dfbf19230.png)
![Screenshot (60)](https://user-images.githubusercontent.com/66817101/224567473-fda83256-4a45-4436-b253-0d04730dbc4f.png)
![Screenshot (61)](https://user-images.githubusercontent.com/66817101/224567476-a551c55e-388f-4709-a25e-0c248b6a2da8.png)
![Screenshot (62)](https://user-images.githubusercontent.com/66817101/224567478-c1d656a1-35d5-429a-a054-d8962a48b3fa.png)
![Screenshot (63)](https://user-images.githubusercontent.com/66817101/224567487-d3cd075b-c896-467d-b9ae-c3b09c3ccf6b.png)
![Screenshot (64)](https://user-images.githubusercontent.com/66817101/224567489-200801a8-f1a1-4edb-b43e-4c4b24ccc50b.png)
![Screenshot (65)](https://user-images.githubusercontent.com/66817101/224567490-28ea2147-f397-4622-b454-1004ac061233.png)
![Screenshot (66)](https://user-images.githubusercontent.com/66817101/224567495-5fdca594-00d4-4d28-8fbf-2a59a6eb783b.png)
![Screenshot (67)](https://user-images.githubusercontent.com/66817101/224567496-0c8483b3-1ea0-4a14-bfe8-4d620141834d.png)
![Screenshot (68)](https://user-images.githubusercontent.com/66817101/224567502-5b303a70-b338-4f17-9121-caa28c24d785.png)
![Screenshot (69)](https://user-images.githubusercontent.com/66817101/224567503-4a006f9f-8c16-43dc-a157-2cd154bb1003.png)
![Screenshot (70)](https://user-images.githubusercontent.com/66817101/224567505-41e63ff1-c21a-46f9-b836-fe001465e208.png)
![Screenshot (71)](https://user-images.githubusercontent.com/66817101/224567506-f0e865cf-522b-45a6-aa6c-882d7b652621.png)
![Screenshot (72)](https://user-images.githubusercontent.com/66817101/224567507-1bf1917a-ed5d-4446-941f-9bf9fa80776f.png)
![Screenshot (73)](https://user-images.githubusercontent.com/66817101/224567508-8871b6f4-d02f-462b-a325-a93a7b0e0ab4.png)
![Screenshot (74)](https://user-images.githubusercontent.com/66817101/224567510-956e275c-c072-4248-a89b-5efa962c3dbe.png)
![Screenshot (75)](https://user-images.githubusercontent.com/66817101/224567513-84710cf3-44b1-449a-b39e-632e38b4f9eb.png)
![Screenshot (76)](https://user-images.githubusercontent.com/66817101/224567515-522ef27d-ce3e-4def-930c-a271fd132d95.png)
![Screenshot (77)](https://user-images.githubusercontent.com/66817101/224567518-6ff09c3c-59c5-4553-8382-8e4924ea8a41.png)
![Screenshot (78)](https://user-images.githubusercontent.com/66817101/224567521-91adb2ca-d9f7-41d5-badd-6930649b3f9c.png)
![Screenshot (79)](https://user-images.githubusercontent.com/66817101/224567525-5f379160-484d-453f-94aa-4dd154205538.png)
![Screenshot (80)](https://user-images.githubusercontent.com/66817101/224567526-7117d7e2-73cc-43b4-a860-9726e7299e0f.png)

