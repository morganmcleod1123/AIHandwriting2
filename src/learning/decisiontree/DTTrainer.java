package learning.decisiontree;

import core.Duple;
import learning.core.Histogram;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DTTrainer<V,L, F, FV extends Comparable<FV>> {
	private ArrayList<Duple<V,L>> baseData;
	private boolean restrictFeatures;
	private Function<ArrayList<Duple<V,L>>, ArrayList<Duple<F,FV>>> allFeatures;
	private BiFunction<V,F,FV> getFeatureValue;
	private Function<FV,FV> successor;
	
	public DTTrainer(ArrayList<Duple<V, L>> data, Function<ArrayList<Duple<V, L>>, ArrayList<Duple<F,FV>>> allFeatures,
					 boolean restrictFeatures, BiFunction<V,F,FV> getFeatureValue, Function<FV,FV> successor) {
		baseData = data;
		this.restrictFeatures = restrictFeatures;
		this.allFeatures = allFeatures;
		this.getFeatureValue = getFeatureValue;
		this.successor = successor;
	}
	
	public DTTrainer(ArrayList<Duple<V, L>> data, Function<ArrayList<Duple<V,L>>, ArrayList<Duple<F,FV>>> allFeatures,
					 BiFunction<V,F,FV> getFeatureValue, Function<FV,FV> successor) {
		this(data, allFeatures, false, getFeatureValue, successor);
	}

	// TODO: Call allFeatures.apply() to get the feature list. Then shuffle the list, retaining
	//  only targetNumber features. Should pass DTTest.testReduced().
	public static <V,L, F, FV  extends Comparable<FV>> ArrayList<Duple<F,FV>>
	reducedFeatures(ArrayList<Duple<V,L>> data, Function<ArrayList<Duple<V, L>>, ArrayList<Duple<F,FV>>> allFeatures,
					int targetNumber) {
		ArrayList<Duple<F, FV>> featureList = allFeatures.apply(data);
		Collections.shuffle(featureList);
		ArrayList<Duple<F, FV>> finalFeatures = new ArrayList<>();
		for(int i = 0; i < targetNumber; i++){
			finalFeatures.add(featureList.remove(0));
		}
		return finalFeatures;
    }
	
	public DecisionTree<V,L,F,FV> train() {
		return train(baseData);
	}

	public static <V,L> int numLabels(ArrayList<Duple<V,L>> data) {
		return data.stream().map(Duple::getSecond).collect(Collectors.toUnmodifiableSet()).size();
	}
	
	private DecisionTree<V,L,F,FV> train(ArrayList<Duple<V,L>> data) {
		if (numLabels(data) == 1) {
			return new DTLeaf<>(data.remove(0).getSecond());
		} else {
			ArrayList<Duple<F,FV>> featureList;
			if(!restrictFeatures){
				featureList = allFeatures.apply(data);
			} else {
				int targetNumber = (int) Math.round(Math.sqrt(data.size()));
				featureList = reducedFeatures(data, allFeatures, targetNumber);
			}
			// Get everything ready for DTInterior call
			double bestCombo = Double.MIN_VALUE;
			Duple<ArrayList<Duple<V,L>>,ArrayList<Duple<V,L>>> bestSplit = new Duple<>(new ArrayList<>(),new ArrayList<>());
			F decisionFeature = null;
			FV maxFeatureValue = null;
			for(Duple<F,FV> feature : featureList){
				Duple<ArrayList<Duple<V,L>>,ArrayList<Duple<V,L>>> splits = splitOn(data, feature.getFirst(),
						feature.getSecond(), getFeatureValue);
				if(gain(data, splits.getFirst(), splits.getSecond()) > bestCombo){
					bestCombo = gain(data, splits.getFirst(), splits.getSecond());
					bestSplit = splits;
					decisionFeature = feature.getFirst();
					maxFeatureValue = feature.getSecond();
				}
			}
			// Return Node or Leaf
			if(bestSplit.getFirst().size() == 0){
				return new DTLeaf<>(mostPopularLabelFrom(bestSplit.getSecond()));
			} else if(bestSplit.getSecond().size() == 0){
				return new DTLeaf<>(mostPopularLabelFrom(bestSplit.getFirst()));
			} else {
				return new DTInterior<>(decisionFeature, maxFeatureValue, train(bestSplit.getFirst()),
						train(bestSplit.getSecond()), getFeatureValue, successor);
			}
		}		
	}

	public static <V,L> L mostPopularLabelFrom(ArrayList<Duple<V, L>> data) {
		Histogram<L> h = new Histogram<>();
		for (Duple<V,L> datum: data) {
			h.bump(datum.getSecond());
		}
		return h.getPluralityWinner();
	}

	// TODO: Generate a new dataset by sampling randomly with replacement.
	// Take data ArrayList and randomly select elements from it and put them into a new Arraylist.
	// The idea of this function is to filter out noise and outliers while giving more importance to
	// elements that occur often.
	public static <V,L> ArrayList<Duple<V,L>> resample(ArrayList<Duple<V,L>> data) {
		ArrayList<Duple<V,L>> resampledList = new ArrayList<>();
		for(int i = 0; i < data.size(); i++){
			Random random = new Random();
			resampledList.add(data.remove(random.nextInt(data.size())));
		}
		return resampledList;
	}

	public static <V,L> double getGini(ArrayList<Duple<V,L>> data) {
		// TODO: Calculate the Gini coefficient:
		//  For each label, calculate its portion of the whole (p_i).
		//  Use of a Histogram<L> for this purpose is recommended.
		//  Gini coefficient is 1 - sum(for all labels i, p_i^2)
		//  Should pass DTTest.testGini().
		Histogram<L> histogram = new Histogram<>();
		ArrayList<L> uniqueLabels = new ArrayList<>();
		for(Duple<V,L> point : data){
			histogram.bump(point.getSecond());
			if(!uniqueLabels.contains(point.getSecond())){
				uniqueLabels.add(point.getSecond());
			}
		}
		double piSum = 0;
		for(L label : uniqueLabels){
			piSum += Math.pow((histogram.getPortionFor(label)), 2.0) ;
		}
		return 1 - piSum;
	}

	public static <V,L> double gain(ArrayList<Duple<V,L>> parent, ArrayList<Duple<V,L>> child1,
									ArrayList<Duple<V,L>> child2) {
		// TODO: Calculate the gain of the split. Add the gini values for the children.
		//  Subtract that sum from the gini value for the parent. Should pass DTTest.testGain().
		return getGini(parent) - (getGini(child1) + getGini(child2));
	}

	public static <V,L, F, FV  extends Comparable<FV>> Duple<ArrayList<Duple<V,L>>,ArrayList<Duple<V,L>>> splitOn
			(ArrayList<Duple<V,L>> data, F feature, FV featureValue, BiFunction<V,F,FV> getFeatureValue) {
		// TODO:
		//  Returns a duple of two new lists of training data.
		//  The first returned list should be everything from this set for which
		//  feature has a value less than or equal to featureValue. The second
		//  returned list should be everything else from this list.
		//  Should pass DTTest.testSplit().
		ArrayList<Duple<V,L>> firstList = new ArrayList<>();
		ArrayList<Duple<V,L>> secondList = new ArrayList<>();
		for(Duple<V,L> point : data){
			int comparison = getFeatureValue.apply(point.getFirst(), feature).compareTo(featureValue);
			if(comparison <= 0){
				firstList.add(point);
			} else {
				secondList.add(point);
			}
		}
		return new Duple<>(firstList, secondList);
	}
}
