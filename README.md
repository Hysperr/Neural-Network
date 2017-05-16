# Neural_Network 
###### C++11 Required ######

#### Back-propagation algorithm for learning in multilayer networks

<pre>
<b>function</b> BACK-PROP-LEARNING(<i>examples</i>, <i>network</i>) <b>returns</b> a neural network
	<b>inputs:</b> <i>examples</i>, a set of examples, each with input vector <b>x</b> and output vector <b>y</b>
		<i>network</i>, a multilayer network with <i>L</i> layers, weights <i>w<sub>i,j</sub></i> activation function <i>g</i>
	<b>local variables</b>: Δ, a vector of errors, indexed by network node
	<b>repeat</b>
		<b>for each</b> weight <i>w<sub>i,j</sub></i> in <i>network</i> <b>do</b>
			<i>w<sub>i,j</sub></i> ← a small random number
		<b>for each</b> example (<b>x</b>, <b>y</b>) <b>in</b> <i>examples</i> <b>do</b>
			<i>/* Propagate the inputs forward to compute the outputs */</i>
			<b>for each</b> node <i>i</i> in the input layer <b>do</b>
				<i>a<sub>i</sub> ← x<sub>i</sub></i>
			<b>for</b> <i>&#x2113;</i> = 2 <b>to</b> <i>L</i> <b>do</b>
				<b>for each</b> node <i>j</i> in layer <i>&#x2113;</i> <b>do</b>
					<i>in<sub>j</sub> ← Σ<sub>i</sub> w<sub>i,j</sub> a<sub>i</sub></i>
					<i>a<sub>j</sub> ← g(in<sub>j</sub>)</i>
			<i>/* Propagate deltas backward from output layer to input layer */</i>
			<b>for each</b> node <i>j</i> in the output layer <b>do</b>
				Δ[i] ← g' (in<sub>i</sub>) Σ<sub>j</sub> w<sub>i,j</sub> Δ[j]
			<b>for</b> &#x2113; = L - 1 <b>to</b> 1 <b>do</b>
				<b>for each</b> node <i>i</i> in layer <i>&#x2113;</i> <b>do</b>
					<i>Δ[i] ← g' (in<sub>i</sub>) Σ<sub>j</sub> w<sub>i,j</sub> Δ[j]</i>
			<i>/* Update every weight in network using deltas */</i>
			<b>for each</b> weight <i>w<sub>i,j</sub></i> in <i>network</i> <b>do</b>
				<i>w<sub>i,j</sub> ← w<sub>i,j</sub> + α x α<sub>i</sub> x Δ[j]</i>
	<b>until</b> some stopping criterion is satisfied
	<b>return</b> <i>network</i>
</pre>

These files are part of a personal attempt to create a Convolutional Neural Network (CNN/ConvNet).
The two classes that make the base neural net are tightly coupled such that they act as a single unit, only separated.
In short, the Node class provides the building blocks to build the NeuralNet class which allows users to create multiple feed-forward multilayered artificial neural networks with backpropagation. From there a linker class will join these such that we have another neural net object where each node is itself a neural net.

## Included Files
* A driver file, digit training file, and test file are provided as a base demonstration. 
* A configuration file provided to tweak your neural net characteristics.

## Function Call Order

1. Start by initializing your constant variables  
   * #### number of input nodes
   * #### number of output nodes
   * #### learning rate
2. Create a `std::map` for hidden layer nodes
   * #### key = layer number _(start from 0, increment by 1)_, value = nodes per layer
3. Call constructor `NeuralNet()`
   * #### The default parameter to include bias nodes is `false`. `true` to activate
4. Call `set_output_identity()`
   * #### pass in a `std::map` to label your output nodes
   * #### key = output node _(start from 0, increment by 1)_, value = output node's identity
5. Insert data using `insert_data()`
   * #### pass in a `std::vector` holding the data
6. Call `forward_propagate()`
   * #### activates and passes data
7. Call `choose_answer()` to have net select its answer
   * #### The default parameter to print neural net's belief values is `false`. `true` to activate
   * #### _If testing after training DO NOT backpropagate!_ Go to step 9
8. Call `back_propagate()`
   * #### trains network by updating weights
9. *If running multiple epochs, remember to clear network!*  


# Congrats! You've Run A Neural Network :D
