package learning.classifiers;

import core.Duple;
import learning.core.Classifier;
import learning.core.Histogram;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.function.ToDoubleBiFunction;

// KnnTest.test() should pass once this is finished.
public class Knn<V, L> implements Classifier<V, L> {
    private ArrayList<Duple<V, L>> data = new ArrayList<>();
    private ToDoubleBiFunction<V, V> distance;
    private int k;

    public Knn(int k, ToDoubleBiFunction<V, V> distance) {
        this.k = k;
        this.distance = distance;
    }

    @Override
    public L classify(V value) {
        Histogram<L> histogram= new Histogram<>();
        ArrayList<Duple<Double,L>> distances = new ArrayList<>();
        for(Duple<V, L> point : data){
            Double dist = distance.applyAsDouble(value, point.getFirst());
            distances.add(new Duple<>(dist, point.getSecond()));
        }
        Collections.sort(distances, Comparator.comparingDouble(Duple::getFirst));
        ArrayList<Duple<Double,L>> topKs = new ArrayList();
        int numKs = Math.min(distances.size(), k);
        for(int i = 0; i < numKs; i++){
            topKs.add(distances.remove(0));
        }
        for(Duple<Double,L> top : topKs){
            histogram.bump(top.getSecond());
        }
        return histogram.getPluralityWinner();
    }

    @Override
    public void train(ArrayList<Duple<V, L>> training) {
        for (Duple<V, L> element : training) {
            if (!data.contains(element)) {
                data.add(element);
            }
        }
    }
}
