package ml;

import java.util.ArrayList;

public class SVM {
	
	
	
	
	public static Vector trainClassifier(ArrayList<Vector> data, int epochs, double C, double rate)
	{
		Vector classifier = new Vector(data.get(0).size());
		
		double t = 0;
		
		for(int epoch = 0; epoch < epochs; epoch++)
		{
			for(Vector x : data)
			{
				double y = x.getLabel();
				double r = rate/(1 + rate*t/C);
				classifier = SVMUpdate(x, classifier, C, r, y);
				t++;
			}
		}
		
		return classifier;
	}
	
	private static Vector SVMUpdate(Vector x, Vector w, double C, double r, double y)
	{
		double[] data = new double[x.size()];
		double transpose = y*w.transpose(x);
		for(int i = 0; i < x.size(); i++)
		{
			if(transpose <= 1)
			{
				data[i] = w.get(i) - r*(w.get(i) - C*y*x.get(i));
			}
			else
			{
				data[i] = w.get(i) - r*w.get(i);
			}
		}
		return new Vector(data);

	}

}
