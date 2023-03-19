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



# TensorFlow implementation of collaborative filtering

In this video, we'll take a look at how you can use TensorFlow to implement the collaborative filtering algorithm. You might be used to thinking of TensorFlow as a tool for building neural networks. And it is. It's a great tool for building neural networks. And it turns out that TensorFlow can also be very hopeful for building other types of learning algorithms as well. Like the collaborative filtering algorithm. One of the reasons I like using TensorFlow for talks like these is that for many applications in order to implement gradient descent, you need to find the derivatives of the cost function, but TensorFlow can automatically figure out for you what are the derivatives of the cost function. All you have to do is implement the cost function and without needing to know any calculus, without needing to take derivatives yourself, you can get TensorFlow with just a few lines of code to compute that derivative term, that can be used to optimize the cost function. 

Let's take a look at how all this works. You might remember this diagram here on the right from course one. This is exactly the diagram that we had looked at when we talked about optimizing w. When we were working through our first linear regression example. And at that time we had set b=0. And so the model was just predicting f(x)=w.x. And we wanted to find the value of w that minimizes the cost function J. So the way we were doing that was via a gradient descent update, which looked like this, where w gets repeatedly updated as w minus the learning rate alpha times the derivative term. If you are updating b as well, this is the expression you will use. But if you said b=0, you just forgo the second update and you keep on performing this gradient descent update until convergence. Sometimes computing this derivative or partial derivative term can be difficult. And it turns out that TensorFlow can help with that. Let's see how. I'm going to use a very simple cost function J=(wx-1) squared. So wx is our simplified f w of x and y is equal to 1. And so this would be the cost function if we had f(x) equals wx,y equals 1 for the one training example that we have, and if we were not optimizing this respect to b. So the gradient descent algorithm will repeat until convergence this update over here. It turns out that if you implement the cost function J over here, TensorFlow can automatically compute for you this derivative term and thereby get gradient descent to work. I'll give you a high level overview of what this code does, w=tf.variable(3.0). Takes the parameter w and initializes it to the value of 3.0. Telling TensorFlow that w is a variable is how we tell it that w is a parameter that we want to optimize. I'm going to set x=1.0, y=1.0, and the learning rate alpha to be equal to 0.01. And let's run gradient dissent for 30 iterations. So in this code will still do for iter in range iterations, so for 30 iterations. And this is the syntax to get TensorFlow to automatically compute the rotors for you. TensorFlow has a feature called a gradient tape. And if you write this with tf our gradient tape as tape f. This is compute f(x) as w*x and compute J as f(x)-y squared. Then by telling TensorFlow how to compute to costJ, and by doing it with the gradient taped syntax as follows, TensorFlow will automatically record the sequence of steps. The sequence of operations needed to compute the costJ. And this is needed to enable automatic differentiation. Next TensorFlow will have saved the sequence of operations in tape, in the gradient tape. And with this syntax, TensorFlow will automatically compute this derivative term, which I'm going to call dJdw. And TensorFlow knows you want to take the derivative respected w. That w is the parameter you want to optimize because you had told it so up here. And because we're also specifying it down here. So now the computer derivatives, finally you can carry out this update by taking w and subtracting from it the learning rate alpha times that derivative term that we just got from up above. TensorFlow variables, tier variables requires special handling. Which is why instead of setting w to be w minus alpha times the derivative in the usual way, we use this assigned add function. But when you get to the practice lab, don't worry about it. We'll give you all the syntax you need in order to implement the collateral filtering algorithm correctly. So notice that with the gradient tape feature of TensorFlow, the main work you need to do is to tell it how to compute the cost function J. And the rest of the syntax causes TensorFlow to automatically figure out for you what is that derivative? And with this TensorFlow we'll start with finding the slope of this, at 3 shown by this dash line. Take a gradient step and update w and compute the derivative again and update w over and over until eventually it gets to the optimal value of w, which is at w equals 1. 

So this procedure allows you to implement gradient descent without ever having to figure out yourself how to compute this derivative term. This is a very powerful feature of TensorFlow called Auto Diff. And some other machine learning packages like pytorch also support Auto Diff. Sometimes you hear people call this Auto Grad. The technically correct term is Auto Diff, and Auto Grad is actually the name of the specific software package for doing automatic differentiation, for taking derivatives automatically. But sometimes if you hear someone refer to Auto Grad, they're just referring to this same concept of automatically taking derivatives. So let's take this and look at how you can implement to collaborative filtering algorithm using Auto Diff. And in fact, once you can compute derivatives automatically, you're not limited to just gradient descent. You can also use a more powerful optimization algorithm, like the adam optimization algorithm. In order to implement the collaborative filtering algorithm TensorFlow, this is the syntax you can use. Let's starts with specifying that the optimizer is keras optimizers adam with learning rate specified here. And then for say, 200 iterations, here's the syntax as before with tf gradient tape, s tape, you need to provide code to compute the value of the cost function J. So recall that in collaborative filtering, the cost function J takes is input parameters x, w, and b as well as the ratings mean normalized. So that's why I'm writing y norm, r(i,j) specifying which values have a rating, number of users or nu in our notation, number of movies or nm in our notation or just num as well as the regularization parameter lambda. And if you can implement this cost function J, then this syntax will cause TensorFlow to figure out the derivatives for you. Then this syntax will cause TensorFlow to record the sequence of operations used to compute the cost. And then by asking it to give you grads equals tape.gradient, this will give you the derivative of the cost function with respect to x, w, and b. 

And finally with the optimizer that we had specified up on top, as the adam optimizer. You can use the optimizer with the gradients that we just computed. And does it function in python is just a function that rearranges the numbers into an appropriate ordering for the applied gradients function. If you are using gradient descent for collateral filtering, recall that the cost function J would be a function of w, b as well as x. And if you are applying gradient descent, you take the partial derivative respect the w. And then update w as follows. And you would also take the partial derivative of this respect to b. And update b as follows. And similarly update the features x as follows. And you repeat until convergence. But as I mentioned earlier with TensorFlow and Auto Diff you're not limited to just gradient descent. You can also use a more powerful optimization algorithm like the adam optimizer. The data set you use in the practice lab is a real data set comprising actual movies rated by actual people. This is the movie lens dataset and it's due to Harper and Konstan. And I hope you enjoy running this algorithm on a real data set of movies, and ratings and see for yourself the results that this algorithm can get. 

So that's it. That's how you can implement the collaborative filtering algorithm in TensorFlow. If you're wondering why do we have to do it this way? <h1> Why couldn't we use a dense layer and then model compiler and model fit? The reason we couldn't use that old recipe is, the collateral filtering algorithm and cost function, it doesn't neatly fit into the dense layer or the other standard neural network layer types of TensorFlow. That's why we had to implement it this other way where we would implement the cost function ourselves. But then use TensorFlow's tools for automatic differentiation, also called Auto Diff.</h1> And use TensorFlow's implementation of the adam optimization algorithm to let it do a lot of the work for us of optimizing the cost function. If the model you have is a sequence of dense neural network layers or other types of layers supported by TensorFlow, and the old implementation recipe of model compound model fit works. But even when it isn't, these tools TensoFlow give you a very effective way to implement other learning algorithms as well. And so I hope you enjoy playing more with the collateral filtering exercise in this week's practice lab. And looks like there's a lot of code and lots of syntax, don't worry about it. Make sure you have what you need to complete that exercise successfully. And in the next video, I'd like to also move on to discuss more of the nuances of collateral filtering and specifically the question of how do you find related items, given one movie, whether other movies similar to this one.

![Screenshot (81)](https://user-images.githubusercontent.com/66817101/226166838-d88b9ebe-1faf-4fa4-bdac-39dea3ecbb98.png)
![Screenshot (82)](https://user-images.githubusercontent.com/66817101/226166843-37d2932b-972c-44e0-a4d3-a34e97175c83.png)
![Screenshot (83)](https://user-images.githubusercontent.com/66817101/226166847-12a1391c-6537-4349-a936-e98f11c6208e.png)
![Screenshot (84)](https://user-images.githubusercontent.com/66817101/226166864-1501d299-a410-4d52-8b1b-3b2aac5e9046.png)
![Screenshot (85)](https://user-images.githubusercontent.com/66817101/226166872-405b9fd5-28c2-458d-95a5-2700abf581d9.png)
![Screenshot (86)](https://user-images.githubusercontent.com/66817101/226166875-3e86f5c4-d226-43d7-954f-9d280500b864.png)
![Screenshot (87)](https://user-images.githubusercontent.com/66817101/226166878-3e6f96fa-548e-4a20-a1a5-6a440488b0e9.png)
![Screenshot (88)](https://user-images.githubusercontent.com/66817101/226166879-543478af-4ea2-46e3-8b7b-74e7a060dfcb.png)
![Screenshot (89)](https://user-images.githubusercontent.com/66817101/226166880-a07ca080-3df9-4e8f-9965-7716e7417899.png)
![Screenshot (90)](https://user-images.githubusercontent.com/66817101/226166882-d5508662-b687-45eb-9bab-0615fcd959dd.png)
![Screenshot (91)](https://user-images.githubusercontent.com/66817101/226166884-24d00fec-f754-46d4-9a5d-32dad65f15ef.png)
![Screenshot (92)](https://user-images.githubusercontent.com/66817101/226166885-e87dafce-f37a-4ddb-a0b6-b59ca7788a8d.png)
![Screenshot (93)](https://user-images.githubusercontent.com/66817101/226166886-b932c999-b891-4a31-a110-9c04ecd99b63.png)
![Screenshot (94)](https://user-images.githubusercontent.com/66817101/226166887-63c82ddf-e816-4b72-af1e-577acb7353df.png)
![Screenshot (95)](https://user-images.githubusercontent.com/66817101/226166889-d9b8d003-5688-43cb-aa04-b92fb2494b33.png)
![Screenshot (96)](https://user-images.githubusercontent.com/66817101/226166891-27538922-ac6c-4485-92cb-df954f6d2d96.png)
![Screenshot (97)](https://user-images.githubusercontent.com/66817101/226166893-38fa2fc9-559f-42d1-8c1a-2da2f644fc3c.png)
![Screenshot (98)](https://user-images.githubusercontent.com/66817101/226166895-09e64d56-4ebd-4999-a26d-33950bc5e399.png)
![Screenshot (99)](https://user-images.githubusercontent.com/66817101/226166896-0b181920-606c-49ea-95f0-87d6fab22533.png)
![Screenshot (100)](https://user-images.githubusercontent.com/66817101/226166897-298b2c2d-91e3-4cc7-8268-e61e899e10de.png)
![Screenshot (101)](https://user-images.githubusercontent.com/66817101/226166901-9d6fa193-4d8f-44d4-9f2d-8469c0e24577.png)
![Screenshot (102)](https://user-images.githubusercontent.com/66817101/226166905-a3c2b4c5-312a-4813-8df9-43ca1bc14e34.png)
![Screenshot (103)](https://user-images.githubusercontent.com/66817101/226166906-0168ce1d-8b26-43a5-9d17-0fceb56dc655.png)
![Screenshot (104)](https://user-images.githubusercontent.com/66817101/226166909-e424cb88-f5ee-452c-a007-afd4723282d4.png)
![Screenshot (105)](https://user-images.githubusercontent.com/66817101/226166914-6a5df6d0-5b98-435d-945f-c3540d55c4ec.png)
![Screenshot (106)](https://user-images.githubusercontent.com/66817101/226166919-78fa5e34-b653-44d4-a325-2bfc6a313bb7.png)
![Screenshot (107)](https://user-images.githubusercontent.com/66817101/226166921-90815157-2dc3-4b3a-bb87-15de4a7861d8.png)
![Screenshot (108)](https://user-images.githubusercontent.com/66817101/226166924-57f3e09c-7bb3-4096-b21e-fd1d4f9cc052.png)
![Screenshot (109)](https://user-images.githubusercontent.com/66817101/226166926-4162106a-3381-4813-8401-a742ad3a918b.png)
![Screenshot (110)](https://user-images.githubusercontent.com/66817101/226166927-060afa02-b0e4-4a26-a6d2-ac02aaa30453.png)
![Screenshot (111)](https://user-images.githubusercontent.com/66817101/226166930-0516fd4d-ae7f-4dc6-a745-1c407a3b324e.png)
![Screenshot (112)](https://user-images.githubusercontent.com/66817101/226166931-eea87999-19fb-4eaa-bad1-92e648183ced.png)
![Screenshot (113)](https://user-images.githubusercontent.com/66817101/226166932-5f2a4617-c8f3-4bda-84ab-ba82c1648bbf.png)
![Screenshot (114)](https://user-images.githubusercontent.com/66817101/226166935-5ea51162-b89b-48cf-b588-bca28730c5fd.png)
![Screenshot (115)](https://user-images.githubusercontent.com/66817101/226166937-83102770-0398-4c45-877f-9c3fd308b7a2.png)
![Screenshot (116)](https://user-images.githubusercontent.com/66817101/226166939-2a29066a-8a06-4a35-8119-885eb4b61c0d.png)
![Screenshot (117)](https://user-images.githubusercontent.com/66817101/226166940-02d8c029-330b-4d0d-ab77-903b3468fde0.png)
![Screenshot (118)](https://user-images.githubusercontent.com/66817101/226166941-2cbc4bb4-fd5d-46b6-a002-b4adb941394b.png)
![Screenshot (119)](https://user-images.githubusercontent.com/66817101/226166942-164587ea-e5c3-47a3-a34f-5ee0cba09dcb.png)
![Screenshot (120)](https://user-images.githubusercontent.com/66817101/226166961-fcdf3985-ce8e-4c3e-af7b-ea6bb4498d6f.png)
![Screenshot (121)](https://user-images.githubusercontent.com/66817101/226166963-093b1947-c0d3-4467-9188-f812ac83c2b7.png)
![Screenshot (122)](https://user-images.githubusercontent.com/66817101/226166964-e7a470c1-74b7-4daa-ba3d-08275a510c25.png)
![Screenshot (123)](https://user-images.githubusercontent.com/66817101/226166973-3dcd6a8f-69ce-41b2-801e-6b04f2cbffcb.png)
![Screenshot (124)](https://user-images.githubusercontent.com/66817101/226166982-92ad789e-77e5-42e1-b117-353a06e1d504.png)
![Screenshot (125)](https://user-images.githubusercontent.com/66817101/226166999-1439cbaa-da6e-483e-9586-06b50f34b0b4.png)
![Screenshot (126)](https://user-images.githubusercontent.com/66817101/226167004-757b6a98-067a-4a9c-9190-871fd92195a3.png)
![Screenshot (127)](https://user-images.githubusercontent.com/66817101/226167011-0dab57bf-0466-4fc5-a0fd-132b22ccaf1e.png)
![Screenshot (128)](https://user-images.githubusercontent.com/66817101/226167012-6332e2bf-cf04-4355-8b06-7c6f01a2828d.png)
![Screenshot (129)](https://user-images.githubusercontent.com/66817101/226167013-d5bf44b3-4cb8-4fc2-81e6-b751a35753f7.png)
![Screenshot (130)](https://user-images.githubusercontent.com/66817101/226167015-568ecd38-252d-4f0b-bf4b-82ab07fff2aa.png)
![Screenshot (131)](https://user-images.githubusercontent.com/66817101/226167019-f2ccd689-665c-4f5d-a310-6b2192202437.png)
![Screenshot (132)](https://user-images.githubusercontent.com/66817101/226167046-9352cb11-1d48-4abe-8e42-e85dbd3b4005.png)
