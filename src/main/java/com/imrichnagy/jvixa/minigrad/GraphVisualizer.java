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
import java.util.*;

import static guru.nidi.graphviz.attribute.Rank.RankType.SAME;
import static guru.nidi.graphviz.model.Factory.*;

public class GraphVisualizer {

    public static void visualize(Value root, String filename) throws IOException {
        visualize(root, filename, null, null);
    }

    public static void visualize(Value root, String filename, Network network, List<Value> inputs) throws IOException {
        MutableGraph g = mutGraph("tree").setDirected(true);
        g.graphAttrs().add(Rank.dir(Rank.RankDir.LEFT_TO_RIGHT));

        ValueGraph vg = buildGraph(root);

        Sequence seq = new Sequence();

        Map<Value, Pair<MutableNode, MutableNode>> nodes = new HashMap<>();
        for (Value node : vg.nodes) {
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

        for (Pair<Value, Value> edge : vg.edges) {
            nodes.get(edge.getLeft()).getRight().addLink(nodes.get(edge.getRight()).getLeft());
        }
        nodes.values().forEach(v -> g.add(v.getLeft(), v.getRight()));

        if (network != null) {
            rankGraph(g, nodes, network, inputs);
        }

        Graphviz.fromGraph(g).render(Format.SVG).toFile(new File(System.getProperty("user.home") + "/" + filename + ".svg"));
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

    private record ValueGraph(List<Value> nodes, List<Pair<Value, Value>> edges) {}

    private static final class Sequence {
        private long value = 0;

        public long next() {
            return value++;
        }
    }

    private static ValueGraph buildGraph(Value root) {
        List<Value> nodes = new ArrayList<>();
        List<Pair<Value, Value>> edges = new ArrayList<>();
        trace(root, nodes, edges);
        return new ValueGraph(nodes, edges);
    }

    private static void trace(Value node, List<Value> nodes, List<Pair<Value, Value>> edges) {
        if (nodes.contains(node)) {
            return;
        }
        nodes.add(node);
        for (Value child : node.children) {
            edges.add(Pair.create(child, node));
            trace(child, nodes, edges);
        }
    }
}
