# Neural_Network
These files are part of a personal attempt to create a convolutional neural network (CNN).
The two classes that make the base neural net are tightly coupled such that they act as a single unit, only separated.
In short, the Node class provides the building blocks to build the NeuralNet class which allows users to create multiple feed-forward multilayered artificial neural networks with backpropagation. From there a linker class will join these such that we have another neural net object where each node is itself a neural net.

## Included Files
* A Driver file, digit training, and tester file are provided as a base demonstration. 
* A configuration file provided to tweak your net characteristics.

## Function Call Order

1. Start by initializing your constant variables  
   * #### number of input nodes
   * #### number of output nodes
   * #### learning rate
2. Create a `std::map` for hidden layer nodes
   * #### key = layer number _(start from 0, increment by 1)_, value = nodes per layer
3. Call constructor `NeuralNet()`
   * #### The default parameter to include bias nodes is `false`. `True` to activate
4. Call `set_output_identity()`
   * #### pass in a `std::map` to label your output nodes
   * #### key = output node _(start from 0, increment by 1)_, value = output node's identity
5. Insert data using `insert_data()`
   * #### pass in a `std::vector` holding the data
6. Call `forward_propagate()`
   * #### activates and passes data
7. Call `choose_answer()` to have net select its answer
   * #### _If testing after training DO NOT backpropagate!_ Go to step 9
8. Call `back_propagate()`
   * #### trains network by updating weights
9. *If running multiple epochs, remember to clear network!*
