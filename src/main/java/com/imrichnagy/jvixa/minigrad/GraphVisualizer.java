package com.imrichnagy.jvixa.minigrad;

import com.imrichnagy.jvixa.minigrad.mlp.Layer;
import com.imrichnagy.jvixa.minigrad.mlp.Network;
import com.imrichnagy.jvixa.minigrad.mlp.Operator;
import com.imrichnagy.jvixa.minigrad.mlp.Value;
import guru.nidi.graphviz.attribute.Color;
import guru.nidi.graphviz.attribute.Rank;
import guru.nidi.graphviz.attribute.Shape;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.MutableGraph;
import guru.nidi.graphviz.model.MutableNode;
import org.graalvm.collections.Pair;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static guru.nidi.graphviz.attribute.Rank.RankType.SAME;
import static guru.nidi.graphviz.model.Factory.*;

public class GraphVisualizer {

    public static void visualize(Value root, String filename) throws IOException {
        visualize(root, filename, true, null, null);
    }

    public static void visualize(Value root, String filename, boolean detached, Network network, List<Value> inputs) throws IOException {
        MutableGraph g = mutGraph("tree").setDirected(true);
        g.graphAttrs().add(Rank.dir(Rank.RankDir.LEFT_TO_RIGHT));

        List<Value> allValues = buildGraph(root);

        if (detached) {
            drawDetached(allValues, g, network, inputs);
        } else {
            draw(allValues, g);
        }

        Graphviz.fromGraph(g).render(Format.SVG).toFile(new File(System.getProperty("user.home") + "/" + filename + ".svg"));
    }

    private static void drawDetached(List<Value> values, MutableGraph g, Network network, List<Value> inputs) {

        Sequence seq = new Sequence();

        Map<Value, Pair<MutableNode, MutableNode>> nodes = new HashMap<>();
        for (Value node : values) {
            MutableNode n1 = mutNode("node" + seq.next())
                    .add(Shape.ELLIPSE)
                    .add("label", node.representation);
            if (node.operator == Operator.CONSTANT) {
                n1.add(Color.RED);
            }
            MutableNode n2 = mutNode("node" + seq.next())
                    .add(Shape.BOX)
                    .add("label", String.format("data: %.4f\n grad: %.4f", node.data, node.gradient));
            n1.addLink(n2);
            nodes.put(node, Pair.create(n1, n2));
        }

        for (Value node : values) {
            for (Value child : node.children) {
                nodes.get(child).getRight().addLink(nodes.get(node).getLeft());
            }
        }

        nodes.values().forEach(v -> g.add(v.getLeft(), v.getRight()));

        if (network != null) {
            rankGraph(g, nodes, network, inputs);
        }
    }

    private static void draw(List<Value> vg, MutableGraph g) {

        Sequence seq = new Sequence();

        Map<Value, MutableNode> nodes = new HashMap<>();
        for (Value node : vg) {
            MutableNode n = mutNode("node" + seq.next())
                    .add(Shape.BOX)
                    .add("label", node.representation + String.format(" | data: %.4f\n grad: %.4f", node.data, node.gradient));
            if (node.operator == Operator.CONSTANT) {
                n.add(Color.RED);
            }
            nodes.put(node, n);
        }

        for (Value node : vg) {
            for (Value child : node.children) {
                nodes.get(child).addLink(nodes.get(node));
            }
        }

        nodes.values().forEach(g::add);
    }

    // Works bad for larger networks
    private static void rankGraph(MutableGraph g, Map<Value, Pair<MutableNode, MutableNode>> nodes, Network network, List<Value> inputs) {

        List<MutableNode> inputNodes = new ArrayList<>();
        for (Value value : inputs) {
            inputNodes.add(nodes.get(value).getLeft());
        }
        g.add(graph().graphAttr().with(Rank.inSubgraph(SAME)).with(inputNodes));

        for (Layer layer : network.layers) {
            List<MutableNode> layerNodes = new ArrayList<>();
            for (Value value : layer.parameters()) {
                layerNodes.add(nodes.get(value).getLeft());
            }
            g.add(graph().graphAttr().with(Rank.inSubgraph(SAME)).with(layerNodes));
        }
    }

    private static final class Sequence {
        private long value = 0;

        public long next() {
            return value++;
        }
    }

    private static List<Value> buildGraph(Value root) {
        List<Value> nodes = new ArrayList<>();
        trace(root, nodes);
        return nodes;
    }

    private static void trace(Value node, List<Value> nodes) {
        if (nodes.contains(node)) {
            return;
        }
        nodes.add(node);
        for (Value child : node.children) {
            trace(child, nodes);
        }
    }
}
