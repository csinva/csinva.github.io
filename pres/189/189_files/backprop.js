var nn;
(function (nn) {
    (function (NodeSection) {
        NodeSection[NodeSection["Input"] = 0] = "Input";
        NodeSection[NodeSection["Output"] = 1] = "Output";
        NodeSection[NodeSection["DerInput"] = 2] = "DerInput";
        NodeSection[NodeSection["DerOutput"] = 3] = "DerOutput";
        NodeSection[NodeSection["DerWeight"] = 4] = "DerWeight";
    })(nn.NodeSection || (nn.NodeSection = {}));
    var NodeSection = nn.NodeSection;
    (function (Formula) {
        Formula[Formula["Input"] = 0] = "Input";
        Formula[Formula["Output"] = 1] = "Output";
        Formula[Formula["DerInput"] = 2] = "DerInput";
        Formula[Formula["DerOutput"] = 3] = "DerOutput";
        Formula[Formula["DerWeight"] = 4] = "DerWeight";
    })(nn.Formula || (nn.Formula = {}));
    var Formula = nn.Formula;
    var Node = (function () {
        function Node(id, cx, cy, activation) {
            this.id = id;
            this.cx = cx;
            this.cy = cy;
            this.activation = activation;
            this.inputs = [];
            this.outputs = [];
            this.bias = 0.1;
            this.inputDer = 0;
            this.accInputDer = 0;
        }
        Node.prototype.updateOutput = function () {
            this.totalInput = this.bias;
            for (var _i = 0, _a = this.inputs; _i < _a.length; _i++) {
                var input = _a[_i];
                this.totalInput += input.weight * input.source.output;
            }
            this.output = this.activation.output(this.totalInput);
            return this.output;
        };
        return Node;
    })();
    nn.Node = Node;
    var Link = (function () {
        function Link(source, dest) {
            this.id = source.id + "-" + dest.id;
            this.source = source;
            this.dest = dest;
            this.weight = Math.random() - 0.5;
            this.errorDer = 0;
            this.accErrorDer = 0;
        }
        return Link;
    })();
    nn.Link = Link;
    function buildNetwork(networkShape, activation, outputActivation) {
        var numLayers = networkShape.length;
        var id = 1;
        var network = [];
        for (var layerIdx = 0; layerIdx < numLayers; layerIdx++) {
            var lastLayer = layerIdx === numLayers - 1;
            var currentLayer = [];
            network.push(currentLayer);
            var numNodes = networkShape[layerIdx];
            for (var i = 0; i < numNodes; i++) {
                var node = new Node(id.toString(), 0, 0, lastLayer ? outputActivation : activation);
                currentLayer.push(node);
                id++;
                if (layerIdx >= 1) {
                    for (var _i = 0, _a = network[layerIdx - 1]; _i < _a.length; _i++) {
                        var prevNode = _a[_i];
                        var link = new Link(prevNode, node);
                        prevNode.outputs.push(link);
                        node.inputs.push(link);
                    }
                }
            }
        }
        return network;
    }
    nn.buildNetwork = buildNetwork;
    function buildAndDrawNetwork(networkShape, activation) {
        var svg = d3.select("#mainsvg");
        var boundingRect = svg.node().getBoundingClientRect();
        var width = boundingRect.width;
        var height = boundingRect.height;
        var numLayers = networkShape.length;
        var verticalScale = d3.scale.ordinal()
            .domain(d3.range(numLayers))
            .rangePoints([height, 0], 1);
        var id = 1;
        var network = [];
        for (var layerIdx = 0; layerIdx < numLayers; layerIdx++) {
            var currentLayer = [];
            network.push(currentLayer);
            var numNodes = networkShape[layerIdx];
            var horizontalScale = d3.scale.ordinal()
                .domain(d3.range(numNodes))
                .rangePoints([0, width], 1);
            for (var i = 0; i < numNodes; i++) {
                var node = new Node(id.toString(), horizontalScale(i), verticalScale(layerIdx), activation);
                currentLayer.push(node);
                id++;
                draw.drawNode(node, svg, layerIdx === 0, layerIdx === numLayers - 1);
                if (layerIdx >= 1) {
                    for (var _i = 0, _a = network[layerIdx - 1]; _i < _a.length; _i++) {
                        var prevNode = _a[_i];
                        var link = new Link(prevNode, node);
                        prevNode.outputs.push(link);
                        node.inputs.push(link);
                        draw.drawLink(link, svg);
                    }
                }
            }
        }
        return network;
    }
    nn.buildAndDrawNetwork = buildAndDrawNetwork;
    function forwardProp(network, inputs) {
        var events = [];
        var inputLayer = network[0];
        if (inputs.length !== inputLayer.length) {
            throw new Error("The number of inputs must match the number of nodes in the input layer");
        }
        for (var i = 0; i < inputLayer.length; i++) {
            var node = inputLayer[i];
            node.output = inputs[i];
            events.push({
                updatedPart: { id: node.id, section: NodeSection.Output },
                involvedParts: [],
                formula: Formula.Output
            });
        }
        for (var layerIdx = 1; layerIdx < network.length; layerIdx++) {
            var currentLayer = network[layerIdx];
            for (var _i = 0; _i < currentLayer.length; _i++) {
                var node = currentLayer[_i];
                var parts = node.inputs.map(function (input) {
                    return {
                        id: input.source.id,
                        section: NodeSection.Output
                    };
                });
                events.push({
                    updatedPart: { id: node.id, section: NodeSection.Input },
                    involvedParts: parts,
                    formula: Formula.Input
                });
                node.updateOutput();
            }
            for (var _a = 0; _a < currentLayer.length; _a++) {
                var node = currentLayer[_a];
                events.push({
                    updatedPart: { id: node.id, section: NodeSection.Output },
                    involvedParts: [{ id: node.id, section: NodeSection.Input }],
                    formula: Formula.Output
                });
            }
        }
        return [network[network.length - 1][0].output, events];
    }
    nn.forwardProp = forwardProp;
    function backProp(network, target, errorFunc) {
        var events = [];
        var outputNode = network[network.length - 1][0];
        outputNode.outputDer = errorFunc.der(outputNode.output, target);
        events.push({
            updatedPart: { id: outputNode.id, section: NodeSection.DerOutput },
            involvedParts: [{ id: outputNode.id, section: NodeSection.Output }],
            formula: null
        });
        for (var layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
            var currentLayer = network[layerIdx];
            for (var _i = 0; _i < currentLayer.length; _i++) {
                var node = currentLayer[_i];
                node.inputDer = node.outputDer * node.activation.der(node.totalInput);
                node.accInputDer += node.inputDer;
                events.push({
                    updatedPart: { id: node.id, section: NodeSection.DerInput },
                    involvedParts: [
                        { id: node.id, section: NodeSection.DerOutput },
                        { id: node.id, section: NodeSection.Input }
                    ],
                    formula: Formula.DerInput
                });
            }
            for (var _a = 0; _a < currentLayer.length; _a++) {
                var node = currentLayer[_a];
                for (var _b = 0, _c = node.inputs; _b < _c.length; _b++) {
                    var input = _c[_b];
                    input.errorDer = node.inputDer * input.source.output;
                    input.accErrorDer += input.errorDer;
                    events.push({
                        updatedPart: { id: input.id, section: NodeSection.DerWeight },
                        involvedParts: [
                            { id: node.id, section: NodeSection.DerInput },
                            { id: input.source.id, section: NodeSection.Output }
                        ],
                        formula: Formula.DerWeight
                    });
                }
            }
            if (layerIdx === 1) {
                continue;
            }
            var prevLayer = network[layerIdx - 1];
            for (var _d = 0; _d < prevLayer.length; _d++) {
                var node = prevLayer[_d];
                node.outputDer = 0;
                for (var _e = 0, _f = node.outputs; _e < _f.length; _e++) {
                    var output = _f[_e];
                    node.outputDer += output.weight * output.dest.inputDer;
                }
                events.push({
                    updatedPart: { id: node.id, section: NodeSection.DerOutput },
                    involvedParts: node.outputs.map(function (output) {
                        return {
                            id: output.dest.id,
                            section: NodeSection.DerInput
                        };
                    }),
                    formula: Formula.DerOutput
                });
            }
        }
        return events;
    }
    nn.backProp = backProp;
    function updateWeights(network, learningRate) {
        for (var layerIdx = 1; layerIdx < network.length; layerIdx++) {
            var currentLayer = network[layerIdx];
            for (var _i = 0; _i < currentLayer.length; _i++) {
                var node = currentLayer[_i];
                node.bias -= learningRate * node.accInputDer;
                node.accInputDer = 0;
                for (var _a = 0, _b = node.inputs; _a < _b.length; _a++) {
                    var input = _b[_a];
                    input.weight -= learningRate * input.accErrorDer;
                    input.accErrorDer = 0;
                }
            }
        }
    }
    nn.updateWeights = updateWeights;
})(nn || (nn = {}));
//# sourceMappingURL=backprop.js.map