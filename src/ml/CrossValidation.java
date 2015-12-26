package ml;


import java.util.ArrayList;

public class CrossValidation {


	/**
	 * Returns the best epoch and learning rate for the given data for perceptron.
	 * 
	 * @return 	{epoch, rate}
	 */
	public static double[] perceptronCV(ArrayList<Vector> data, int fold, int[] epochs, double[] rates)
	{
		int size = data.size();
		int bound = (int) (size/fold + Math.ceil((size%fold)/(fold*1.0)));
		ArrayList<ArrayList<Vector>> cross = createCrossValidationSet(data, size, bound);
		
		double maxEpoch = 0; double maxRate = 0; double maxAccuracy = 0;
		
		for(int epoch : epochs)
		{
			for(double rate : rates)
			{
				double accuracy = 0;

				//Looping through sets chosen to stay out
				for(int i = 0; i < fold; i++)
				{					

					ArrayList<Vector> trainSet = new ArrayList<Vector>();
					//Collect the training sets
					for(int j = 0; j < fold; j++)
					{
						if(i == j) continue;//Skip the "testing set"
						trainSet.addAll(cross.get(j));
					}
					//Compute the weight vector for the training sets
					Vector w = Perceptron.trainClassifier(data, epoch, rate);
					//Compute accuracy of the testing set on the training sets
					accuracy += collectAccuracy(cross.get(i), w);			
				}
				//Get the statistical accuracy
				accuracy /= fold;

				//Keep track of the best hyperparameters
				if(accuracy > maxAccuracy)
				{
					maxAccuracy = accuracy;
					maxEpoch = epoch;
					maxRate = rate;
				}
			}
		}
		
		return new double[] {maxEpoch, maxRate};	

	}
	
	public static double[] averagePerceptronCV(ArrayList<Vector> data, int fold, int[] epochs, double[] rates)
	{
		int size = data.size();
		int bound = (int) (size/fold + Math.ceil((size%fold)/(fold*1.0)));
		ArrayList<ArrayList<Vector>> cross = createCrossValidationSet(data, size, bound);
		
		double maxEpoch = 0; double maxRate = 0; double maxAccuracy = 0;
		
		for(int epoch : epochs)
		{
			for(double rate : rates)
			{
				double accuracy = 0;

				//Looping through sets chosen to stay out
				for(int i = 0; i < fold; i++)
				{					

					ArrayList<Vector> trainSet = new ArrayList<Vector>();
					//Collect the training sets
					for(int j = 0; j < fold; j++)
					{
						if(i == j) continue;//Skip the "testing set"
						trainSet.addAll(cross.get(j));
					}
					//Compute the weight vector for the training sets
					Vector w = Perceptron.trainAverageClassifier(data, epoch, rate);
					//Compute accuracy of the testing set on the training sets
					accuracy += collectAccuracy(cross.get(i), w);			
				}
				//Get the statistical accuracy
				accuracy /= fold;

				//Keep track of the best hyperparameters
				if(accuracy > maxAccuracy)
				{
					maxAccuracy = accuracy;
					maxEpoch = epoch;
					maxRate = rate;
				}
			}
		}
		
		return new double[] {maxEpoch, maxRate};	

	}
	
	
	public static double[] supportVectorCV(ArrayList<Vector> data, int fold, double[] C, double[] rates)
	{
		int size = data.size();
		int bound = (int) (size/fold + Math.ceil((size%fold)/(fold*1.0)));
		ArrayList<ArrayList<Vector>> cross = createCrossValidationSet(data, size, bound);
		int epoch = 200;
		double maxC = 0; double maxRate = 0; double maxAccuracy = 0;
		
		for(double c : C)
		{
			for(double rate : rates)
			{
				double accuracy = 0;

				//Looping through sets chosen to stay out
				for(int i = 0; i < fold; i++)
				{					

					ArrayList<Vector> trainSet = new ArrayList<Vector>();
					//Collect the training sets
					for(int j = 0; j < fold; j++)
					{
						if(i == j) continue;//Skip the "testing set"
						trainSet.addAll(cross.get(j));
					}
					//Compute the weight vector for the training sets
					Vector w = SVM.trainClassifier(data, epoch, c, rate);
					//Compute accuracy of the testing set on the training sets
					accuracy += collectAccuracy(cross.get(i), w);			
				}
				//Get the statistical accuracy
				accuracy /= fold;

				//Keep track of the best hyperparameters
				if(accuracy > maxAccuracy)
				{
					maxAccuracy = accuracy;
					maxC = c;
					maxRate = rate;
				}
			}
		}
		
		return new double[] {maxC, maxRate};	

	}
	
	/**
	 * Collects the accuracy over an entire test set on the classifier
	 * @param test
	 * @param classifier
	 * @return
	 */
	private static double collectAccuracy(ArrayList<Vector> test, Vector classifier)
	{
		double accuracy = 0;
		double size = 0;
		
		for(Vector x : test)
		{
			double y = x.getLabel();
			if(y*classifier.transpose(x) > 0)
				accuracy++;
			size++;
		}
		
		return accuracy/size;
	}

	private static ArrayList<ArrayList<Vector>> createCrossValidationSet(ArrayList<Vector> data, double size, int bound)
	{
		ArrayList<ArrayList<Vector>> cross = new ArrayList<ArrayList<Vector>>();
		for(int i = 0; i < size;)
		{
			ArrayList<Vector> list = new ArrayList<Vector>();
			for(int j = i; j < i + bound; j++)
			{
				if(j == size)
					break;
				list.add(data.get(j));
			}
			i += bound;
			cross.add(list);
		}
		return cross;
	}





	private static double[] crossValidation(ArrayList<Vector> data, int fold, double[] C, double[] Rho, double[] Sigma2, boolean verbatim)
	{
		int size = data.size();
		int bound = (int) (size/fold + Math.ceil((size%fold)/(fold*1.0)));
		ArrayList<ArrayList<Vector>> cross = new ArrayList<ArrayList<Vector>>(fold);
		int epoch = 10;



		//Max Accuracy   Best Rho         Best C		   Best Sigma2
		double maxA = 0; double maxP = 0; double maxC = 0; double maxS = 0;

		for(double sigma2 : Sigma2)
		{
			//Loop through all the C hyperparameters
			for(double c : C)
			{
				//Loop through all the rho hyperparameters
				for(double rho : Rho)
				{
					double accuracy = 0;

					//Looping through sets chosen to stay out
					for(int i = 0; i < fold; i++)
					{					

						ArrayList<Vector> trainSet = new ArrayList<Vector>();
						//Collect the training sets
						for(int j = 0; j < fold; j++)
						{
							if(i == j) continue;//Skip the "testing set"
							trainSet.addAll(cross.get(j));
						}
						//Compute the weight vector for the training sets
						//Vector w = SGD(trainSet, rho, c, sigma2, epoch);
						//Compute accuracy of the testing set on the training sets
						//accuracy += collectAccuracy(cross.get(i), w);			
					}
					//Get the statistical accuracy
					accuracy /= fold;

					//Keep track of the best hyperparameters
					if(accuracy > maxA)
					{
						maxA = accuracy;
						maxC = c;
						maxP = rho;
						maxS = sigma2;
					}
					if(verbatim)
						System.out.println(accuracy + "\t" + c + "\t" + rho);
				}
			}	
		}


		System.out.println("--- BEST C: " + maxC + "\tBEST RHO: " + maxP + "\tBEST SIGMA2: " + maxS + "\t WITH ACCURACY: " + maxA);
		double[] ret = {maxP, maxC, maxS};
		return ret;
	}
}
