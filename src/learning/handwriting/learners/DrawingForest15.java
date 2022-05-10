package learning.handwriting.learners;

import core.Duple;
import learning.decisiontree.RandomForest;
import learning.handwriting.core.Drawing;
import learning.handwriting.core.DrawingPoint;
import learning.handwriting.core.PixelUse;
import learning.handwriting.core.PixelUsePattern;
import learning.handwriting.gui.PixelUseVisualizer;
import learning.handwriting.gui.PixelUser;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Optional;

public class DrawingForest15 extends RandomForest<Drawing,String,DrawingPoint, PixelUse> implements PixelUser {
	private int width, height;

	public DrawingForest15() {
		super(15, Drawing::allFeatures, Drawing::getFeatureValue, PixelUse::successor);
	}

	@Override
	public void train(ArrayList<Duple<Drawing, String>> data) {
		this.width = data.get(0).getFirst().getWidth();
		this.height = data.get(0).getFirst().getHeight();
		super.train(data);
	}

	@Override
	public Optional<JPanel> getVisualization() {
		return Optional.of(new PixelUseVisualizer(this));
	}

	@Override
	public PixelUsePattern getPixelUse(String label) {
		PixelUsePattern pixels = new PixelUsePattern(this.width, this.height);
		visualize(label, pixels);
		return pixels;
	}
}
