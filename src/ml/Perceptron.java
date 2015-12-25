package ml;

import java.util.*;

/**
 * 
 * @author Maks Cegielski-Johnson
 *
 */
public class Perceptron {
	
	
	/**
	 * Trains Vanilla Perceptron assuming labels are {-1,1}, non-string.
	 * @param data
	 */
	public static Vector trainClassifier(ArrayList<Vector> data, int epochs, double rate)
	{
		int data_dimension = data.get(0).size();
		
		//Initialize a random weight vector.
		Vector weight = new Vector(data_dimension);
		
		//Loop through the data 'epochs' times. 
		for(int epoch = 0; epoch < epochs; epoch++)
		{
			//Shuffle the data for each epoch.
			data = Util.shuffle(data);
			//Loop through each example
			for(Vector x : data)
			{
				double y = x.getLabel();
				//Check if we made a mistake, if so we update the weight vector.
				if(y*weight.transpose(x) <= 0)
				{
					weight = perceptronUpdate(x, weight, rate, y);
				}
			}
		}
		
		return weight;
	}
	
	/**
	 * Trains Vanilla Perceptron assuming labels are {-1,1}, assuming that we are training for a specific label 'label'
	 * @param data
	 */
	public static void trainClassifier(ArrayList<Vector> data, int epochs, double rate, String label)
	{
		int data_dimension = data.get(0).size();
		
		//Initialize a random weight vector.
		Vector weight = new Vector(data_dimension);
		
		//Loop through the data 'epochs' times. 
		for(int epoch = 0; epoch < epochs; epoch++)
		{
			//Shuffle the data for each epoch.
			data = Util.shuffle(data);
			//Loop through each example
			for(Vector x : data)
			{
				double y = x.getLabel(label);
				//Check if we made a mistake, if so we update the weight vector.
				if(y*weight.transpose(x) <= 0)
				{
					weight = perceptronUpdate(x, weight, rate, y);
				}
			}
		}
	}
	
	public static double predictLabel(Vector weight, Vector example)
	{
		return Math.signum(weight.transpose(example));
	}
	
	/**
	 * Computes the perceptron update, returning the updated weight vector. 
	 * The update only occurs when there is an error while learning. 
	 * 
	 * Recall the update is w = w + r*y*x.
	 * 
	 * @param x
	 * @param w
	 * @param rate
	 * @return
	 */
	private static Vector perceptronUpdate(Vector x, Vector w, double rate, double y)
	{
		double[] data = new double[x.size()];
		for(int i = 0; i < x.size(); i++)
		{
			data[i] = w.get(i) + rate*y*x.get(i);
		}
		return new Vector(data);
	}

}
