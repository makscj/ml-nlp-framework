package ml;

import java.util.Random;

public class Vector {

	
	private double[] data;
	
	private double double_label;
	
	private String string_label;
	
	/**
	 * Assumes that the bias is already contained in the data array, in position 0.
	 * 
	 * @param _data
	 * @param label
	 */
	public Vector(double[] _data, double label)
	{
		data = _data;
		double_label = label;
	}
	
	/**
	 * Assumes that the bias is already contained in the data array, in position 0.
	 * @param _data
	 * @param label
	 */
	public Vector(double[] _data, String label)
	{
		data = _data;
		string_label = label;
	}
	
	/**
	 * Creates a Vector with random values between 0 and 1. 
	 * String label will be null. 
	 * Double label will be zero. Both default values. 
	 * @param size
	 */
	public Vector(int size)
	{
		data = new double[size];
		data[0] = 1;
		Random rng = new Random();
		for(int i = 1; i < size; i++)
		{
			data[i] = rng.nextDouble();
		}
	}
	
	/**
	 * Create a vector without a specific label. Primarily used for weight vectors.
	 * @param _data
	 */
	public Vector(double[] _data)
	{
		data = _data;
	}
	
	/**
	 * Returns the numeric label.
	 * @return
	 */
	public double getLabel()
	{
		return double_label;
	}
	
	/**
	 * Returns 1 if the label matches the 'other' string, returns -1 if they are different.
	 * @param other
	 * @return
	 */
	public double getLabel(String other)
	{
		return other.equals(string_label) ? 1 : -1;
	}
	
	/**
	 * Returns the String representation of the label. Null if there is no String label.
	 * @return
	 */
	public String getStringLabel()
	{
		return string_label;
	}
	
	/**
	 * Returns the transpose of this Vector with another Vector. 
	 * @param other
	 * @return
	 */
	public double transpose(Vector other)
	{
		double sum = 0;
		
		for(int i = 0; i < other.size(); i++)
		{
			sum += other.get(i)*data[i];
		}
		
		return sum;
		
	}
	
	/**
	 * Returns the value at the position in the data array.
	 * @param position
	 * @return
	 */
	public double get(int position)
	{
		return data[position];
	}
	
	/**
	 * Returns the size of the data array.
	 * @return
	 */
	public int size()
	{
		return data.length;
	}
	
}
