package learning.decisiontree;

import core.Duple;
import learning.handwriting.core.Drawing;
import learning.handwriting.core.DrawingPoint;
import learning.handwriting.core.PixelUse;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class DTTest {
    Drawing d1 = new Drawing("3|3|XXX|OOO|XXO");
    Drawing d2 = new Drawing("3|3|XOX|OXO|OXX");
    Drawing d3 = new Drawing("3|3|OXO|OXO|XOX");
    Drawing d4 = new Drawing("3|3|XOX|XOX|XOX");
    Drawing d5 = new Drawing("3|3|OOO|XXX|OOO");
    ArrayList<Duple<Drawing,String>> dtMain, dt1, dt2;

    @Before
    public void setup() {
        dtMain = new ArrayList<>();
        dtMain.add(new Duple<>(d1, "A"));
        dtMain.add(new Duple<>(d2, "A"));
        dtMain.add(new Duple<>(d3, "B"));
        dtMain.add(new Duple<>(d4, "B"));
        dtMain.add(new Duple<>(d5, "B"));

        dt1 = new ArrayList<>();
        dt1.add(new Duple<>(d1, "A"));
        dt1.add(new Duple<>(d3, "B"));
        dt1.add(new Duple<>(d5, "B"));

        dt2 = new ArrayList<>();
        dt2.add(new Duple<>(d2, "A"));
        dt2.add(new Duple<>(d4, "B"));
    }

    @Test
    public void testInterior() {
        DTLeaf<Drawing, String, DrawingPoint, PixelUse> left = new DTLeaf<>("left");
        DTLeaf<Drawing, String, DrawingPoint, PixelUse> right = new DTLeaf<>("right");
        DTInterior<Drawing, String, DrawingPoint, PixelUse> nodeO = new DTInterior<>(new DrawingPoint(0, 1), PixelUse.ON, left, right, Drawing::getFeatureValue, PixelUse::successor);
        DTInterior<Drawing, String, DrawingPoint, PixelUse> nodeX = new DTInterior<>(new DrawingPoint(1, 1), PixelUse.ON, left, right, Drawing::getFeatureValue, PixelUse::successor);

        assertEquals(left.getLabel(), nodeX.classify(d5));
        assertEquals(right.getLabel(), nodeO.classify(d5));
    }

    @Test
    public void testGini() {
        assertEquals(.48, DTTrainer.getGini(dtMain), 0.001);
    }

    @Test
    public void testSplit() {
        Duple<ArrayList<Duple<Drawing,String>>,ArrayList<Duple<Drawing,String>>> splits = DTTrainer.splitOn(dtMain, new DrawingPoint(1, 1), PixelUse.ON, Drawing::getFeatureValue);
        assertFalse(isInside(d1, "A", splits.getFirst()));
        assertTrue(isInside(d1, "A", splits.getSecond()));

        assertTrue(isInside(d2, "A", splits.getFirst()));
        assertFalse(isInside(d2, "A", splits.getSecond()));

        assertTrue(isInside(d3, "B", splits.getFirst()));
        assertFalse(isInside(d3, "B", splits.getSecond()));

        assertFalse(isInside(d4, "B", splits.getFirst()));
        assertTrue(isInside(d4, "B", splits.getSecond()));

        assertTrue(isInside(d5, "B", splits.getFirst()));
        assertFalse(isInside(d5, "B", splits.getSecond()));
    }

    public boolean isInside(Drawing d, String label, ArrayList<Duple<Drawing,String>> samples) {
        for (Duple<Drawing, String> sample: samples) {
            if (sample.getSecond().equals(label) && sample.getFirst().equals(d)) {
                return true;
            }
        }
        return false;
    }

    @Test
    public void testGain() {
        assertEquals(-.46444, DTTrainer.gain(dtMain, dt1, dt2), .001);
    }

    @Test
    public void testTrain() throws InterruptedException {
        DTTrainer<Drawing, String, DrawingPoint, PixelUse> trainer = new DTTrainer<>(dtMain, Drawing::allFeatures,
                Drawing::getFeatureValue, PixelUse::successor);
        DecisionTree<Drawing, String, DrawingPoint, PixelUse> tree = trainer.train();
        assertEquals("A", tree.classify(d1));
        assertEquals("A", tree.classify(d2));
        assertEquals("B", tree.classify(d3));
        assertEquals("B", tree.classify(d4));
        assertEquals("B", tree.classify(d5));
    }

    @Test
    public void testReduced() {
        ArrayList<Duple<DrawingPoint,PixelUse>> features = Drawing.allFeatures(dtMain);
        int height = dtMain.get(0).getFirst().getHeight();
        ArrayList<Duple<DrawingPoint,PixelUse>> reduced = DTTrainer.reducedFeatures(dtMain, Drawing::allFeatures, height);

        assertEquals(height, reduced.size());
        for (Duple<DrawingPoint,PixelUse> feature: reduced) {
            assertTrue(features.contains(feature));
        }
    }
    @Test
    public void testResample() {
        ArrayList<Duple<Drawing,String>> resampled = DTTrainer.resample(dtMain);
        assertEquals(resampled.size(), dtMain.size());
        for (Duple<Drawing,String> example: resampled) {
            assert dtMain.contains(example);
        }
    }
}
