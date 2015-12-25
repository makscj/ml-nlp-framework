package ml;

import java.util.ArrayList;

public class Util {
	
	/**
	 * Shuffles the data set, which is an ArrayList of vectors.
	 * @param array
	 * @return
	 */
	public static ArrayList<Vector> shuffle(ArrayList<Vector> array)
	{
		int index = array.size()-1;
		while(index>1)
		{
			int randomIndex = (int)(Math.random()*index);
			Vector temp = array.get(index);
			array.set(index--,array.get(randomIndex));
			array.set(randomIndex,temp);
		}
		return array;
	}

}
