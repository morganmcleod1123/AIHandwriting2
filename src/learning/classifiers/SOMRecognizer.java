package learning.classifiers;

import core.Duple;
import learning.core.Classifier;
import learning.som.SOMPoint;
import learning.som.SelfOrgMap;
import learning.som.WeightedAverager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.function.Supplier;
import java.util.function.ToDoubleBiFunction;

public class SOMRecognizer<V, L> implements Classifier<V, L> {
    private L[][] labels;
    private SelfOrgMap<V> som;
    private ToDoubleBiFunction<V, V> distance;

    public final static int K = 11;

    public SOMRecognizer(int mapSide, Supplier<V> makeDefault, ToDoubleBiFunction<V, V> distance, WeightedAverager<V> averager) {
        this.distance = distance;
        som = new SelfOrgMap<V>(mapSide, makeDefault, distance, averager);
        labels = (L[][])new Object[mapSide][mapSide];
    }

    @Override
    public void train(ArrayList<Duple<V, L>> data) {
        double prog = 0.0;
        ArrayList<Duple<V,L>> shuffleCopy = new ArrayList<>(data);
        Collections.shuffle(shuffleCopy);
        for (int i = 0; i < shuffleCopy.size(); i++) {
            Duple<V, L> sample = shuffleCopy.get(i);
            som.train(sample.getFirst());
            prog += 1.0 / (2.0 * shuffleCopy.size());
        }

        for (int x = 0; x < som.getMapWidth(); x++) {
            for (int y = 0; y < som.getMapHeight(); y++) {
                labels[x][y] = findLabelFor(som.getNode(x, y), K, shuffleCopy, distance);
                prog += 1.0 / (2.0 * som.getMapHeight() * som.getMapWidth());
            }
        }
    }

    // TODO: Perform a k-nearest-neighbor retrieval to return the label that
    //  best matches the current node.
    public static <V,L> L findLabelFor(V currentNode, int k, ArrayList<Duple<V, L>> allSamples, ToDoubleBiFunction<V, V> distance) {
        Knn<V, L> knn = new Knn<>(k, distance);
        knn.train(allSamples);
        return knn.classify(currentNode);
    }

    @Override
    public L classify(V d) {
        SOMPoint where = som.bestFor(d);
        return labels[where.x()][where.y()];
    }

    public SelfOrgMap<V> getSOM() {return som;}
}
