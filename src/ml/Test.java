package ml;



import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Test {
	
	
	public static void main(String[] args)
	{
		ArrayList<Vector> oldtrain = new ArrayList<Vector>();
		ArrayList<Vector> oldTest = new ArrayList<Vector>();
		ArrayList<Vector> astroTrain = new ArrayList<Vector>();
		ArrayList<Vector> astroTest = new ArrayList<Vector>();
		ArrayList<Vector> astroTrainScaled = new ArrayList<Vector>();
		ArrayList<Vector> astroTestScaled = new ArrayList<Vector>();

		try {
			oldtrain = readOldData("./src/train0.10", 10);
			oldTest = readOldData("./src/test0.10", 10);
			astroTrain = readAstros("./src/original/train");
			astroTest = readAstros("./src/original/test");
			astroTrainScaled = readAstros("./src/scaled/train");
			astroTestScaled = readAstros("./src/scaled/test");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		testPerceptron(astroTrain, astroTest);
	}
	
	
	private static void testPerceptron(ArrayList<Vector> training, ArrayList<Vector> testing)
	{
		System.out.println("------------- Perceptron ------------------");
		int[] epochs = new int[] {10,20,30,50,100,150};
		double[] rates = new double[] {1,0.1,0.01,0.001,0.0001};
		
		double[] hyperparameters = CrossValidation.perceptronCV(training, 10, epochs, rates);
		
		System.out.println("Best epoch: " + hyperparameters[0] +"\tBest rate: " + hyperparameters[1]);
		
		Vector classifier = Perceptron.trainClassifier(training,(int)hyperparameters[0],hyperparameters[1]);
		
		double correct = 0;
		double total = 0;
		
		for(Vector example : testing)
		{
			if(Perceptron.predictLabel(classifier, example) == example.getLabel())			
				correct++;
			total++;
		}
		
		double accuracy = correct/total;
		
		System.out.println("Accuracy on the test set is " + accuracy);
		System.out.println("-------------------------------------------");
	}
	
	
	
	
	private static ArrayList<Vector> readOldData(String filename, int dimension) throws IOException
	{
		ArrayList<Vector> ret = new ArrayList<Vector>();

		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = br.readLine();
		while(line != null)
		{
			String[] pieces = line.split(" ");
			double[] data = new double[dimension+1];
			double label = 0;
			//Vector example = new Vector(dimension+1);
			for(int i = 0; i < pieces.length; i++)
			{				
				if(i == 0)
					label = Integer.parseInt(pieces[i]);
				else
					data[Integer.parseInt(pieces[i].split(":")[0])] = Double.parseDouble(pieces[i].split(":")[1]);
			}
			ret.add(new Vector(data,label));
			line = br.readLine();
		}
		br.close();		
		return ret;
	}


	private static ArrayList<Vector> readAstros(String filename) throws IOException
	{
		ArrayList<Vector> ret = new ArrayList<Vector>();

		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = br.readLine();
		while(line != null)
		{
			String[] pieces = line.split(" ");
			//Dimension will be 4 + bias
			double[] data = new double[5];
			data[0] = 1;
			double label = 0;
			for(int i = 0; i < pieces.length; i++)
			{							
				if(i == 0)
					label = pieces[i].equals("0") ? -1 : 1;				
				else
					data[Integer.parseInt(pieces[i].split(":")[0])] = Double.parseDouble(pieces[i].split(":")[1]);
			}
			ret.add(new Vector(data, label));
			line = br.readLine();
		}
		br.close();		
		return ret;
	}
}
