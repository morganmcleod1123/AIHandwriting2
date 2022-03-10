package learning.decisiontree;

import core.Duple;
import learning.core.Updateable;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;

public class DTInterior<V, L, F, FV extends Comparable<FV>> implements DecisionTree<V,L,F,FV> {
	private DecisionTree<V,L,F,FV> left, right;
	private F decisionFeature;
	private FV maxFeatureValue;
	private BiFunction<V,F,FV> getFeatureValue;
	private Function<FV,FV> successor;
	
	public DTInterior(F decisionFeature, FV maxFeatureValue, DecisionTree<V,L,F,FV> left, DecisionTree<V,L,F,FV> right,
					  BiFunction<V,F,FV> getFeatureValue, Function<FV,FV> successor) {
		this.left = left;
		this.right = right;
		this.decisionFeature = decisionFeature;
		this.maxFeatureValue = maxFeatureValue;
		this.getFeatureValue = getFeatureValue;
		this.successor = successor;
	}

	@Override
	public L classify(V v) {
		// Questions:
		// How to do value comparison?
		// TODO: If the targeted decisionFeature is less than or equal to the maxFeatureValue, ask the left subtree.
		//       Otherwise, ask the right subtree.
		//       DTTest.testInterior() should pass when this works.
		int comparison = getFeatureValue.apply(v, decisionFeature).compareTo(maxFeatureValue);
		if(comparison <= 0){
			return left.classify(v);
		} else {
			return right.classify(v);
		}
	}

	@Override
	public void addAllLabels(Set<L> labels) {
		left.addAllLabels(labels);
		right.addAllLabels(labels);
	}

	@Override
	public <D extends Updateable<Duple<F,FV>>> void visualize(L label, ArrayList<Duple<F, FV>> features, D drawable) {
		ArrayList<Duple<F,FV>> leftData = new ArrayList<>(features);
		leftData.add(new Duple<>(decisionFeature, maxFeatureValue));
		left.visualize(label, leftData, drawable);

		ArrayList<Duple<F,FV>> rightData = new ArrayList<>(features);
		rightData.add(new Duple<F,FV>(decisionFeature, successor.apply(maxFeatureValue)));
		right.visualize(label, rightData, drawable);
	}

	@Override
	public String toStringHelp(int numTabs) {
		StringBuilder sb = new StringBuilder();
		sb.append(tabs(numTabs));
		sb.append("DTInterior: (").append(maxFeatureValue).append(")\n");
		sb.append(left.toStringHelp(numTabs + 1));
		sb.append(right.toStringHelp(numTabs + 1));
		return sb.toString();
	}

	@Override
	public String toString() {
		return toStringHelp(0);
	}

	public boolean equals(Object other) {
		return this.toString().equals(other.toString());
	}
}
