var draw;
(function (draw) {
    var RECT_WIDTH = 50;
    var RECT_HEIGHT = 80;
    var PADDING_TEXT_BOX = 3;
    var NUM_EXAMPLES = 100;
    var TARGET_NOISE = 1;
    var LEARNING_RATE = 5;
    function shorterDist(x1, y1, x2, y2, delta, isPercentage) {
        if (isPercentage === void 0) { isPercentage = false; }
        var angle = Math.atan((y1 - y2) / (x2 - x1));
        if (angle < 0) {
            angle += Math.PI;
        }
        var dist = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y1 - y2, 2));
        var newDist = isPercentage ? dist * delta : dist - delta;
        var endX = x1 + newDist * Math.cos(angle);
        var endY = y1 - newDist * Math.sin(angle);
        return [endX, endY];
    }
    function drawBox(container, className, source, dest, midX, midY, text, subText) {
        var textBox = container.append("g").attr({
            "class": "textbox " + className,
            "id": "weight" + source.id + "-" + dest.id,
            "transform": "translate(" + midX + "," + midY + ")",
            "opacity": 0
        });
        var rectBox = textBox.append("rect").classed(className, true);
        var textSvg = textBox.append("text").text(text);
        if (subText != null) {
            textSvg.append("tspan").text(subText).attr("baseline-shift", "sub");
        }
        textSvg.attr({
            x: 0,
            y: 0,
            "text-anchor": "middle"
        });
        var bBox = textSvg.node().getBBox();
        rectBox.attr({
            x: bBox.x - PADDING_TEXT_BOX,
            y: bBox.y - PADDING_TEXT_BOX,
            width: bBox.width + PADDING_TEXT_BOX * 2,
            height: bBox.height + PADDING_TEXT_BOX * 2
        });
    }
    draw.drawBox = drawBox;
    function drawLink(link, container) {
        var source = link.source;
        var dest = link.dest;
        var line = container.append("line")
            .style("opacity", 0);
        var _a = shorterDist(source.cx, source.cy - RECT_HEIGHT / 2, dest.cx, dest.cy + RECT_HEIGHT / 2, 10), endX = _a[0], endY = _a[1];
        line.attr({
            id: "link" + source.id + "-" + dest.id,
            x1: source.cx,
            y1: source.cy - RECT_HEIGHT / 2,
            x2: endX,
            y2: endY,
            "marker-end": "url(#markerArrow)"
        });
        var _b = shorterDist(source.cx, source.cy - RECT_HEIGHT / 2, dest.cx, dest.cy + RECT_HEIGHT / 2, 0.3, true), midX = _b[0], midY = _b[1];
        drawBox(container, "weight", source, dest, midX, midY - 24, "w", link.id.replace("-", ""));
        drawBox(container, "DerWeight", source, dest, midX, midY, "dE/dw");
    }
    draw.drawLink = drawLink;
    function drawErrorAndTarget(network) {
        var outputNode = network[network.length - 1][0];
        var inputNode = network[0][network[0].length - 1];
        var svg = d3.select("#mainsvg").append("g")
            .attr("class", "error")
            .style("opacity", 0);
        var deltaRight = 110;
        var y = outputNode.cy - RECT_HEIGHT / 2 + RECT_HEIGHT / 8;
        svg.append("circle").attr({
            "cx": outputNode.cx + deltaRight,
            "cy": y,
            "r": 10
        });
        svg.append("text").attr({
            "x": outputNode.cx + deltaRight,
            "y": y + 3,
            "text-anchor": "middle"
        }).text("E");
        svg.append("line").attr({
            "x1": outputNode.cx + RECT_WIDTH / 2,
            "y1": y,
            "x2": outputNode.cx + deltaRight - 20,
            "y2": y,
            "marker-end": "url(#markerArrow)"
        });
        svg.append("rect").attr({
            "x": inputNode.cx + deltaRight - RECT_WIDTH / 2,
            "y": inputNode.cy - RECT_HEIGHT / 2,
            width: RECT_WIDTH,
            height: RECT_HEIGHT / 4,
            "class": "target"
        });
        var text = svg.append("text").attr({
            "x": inputNode.cx + deltaRight,
            "y": inputNode.cy - RECT_HEIGHT / 2 + RECT_HEIGHT / 8,
            "text-anchor": "middle"
        }).text("y");
        text.append("tspan").text("target").attr("baseline-shift", "sub");
        svg.append("line").attr({
            "x1": inputNode.cx + deltaRight,
            "y1": inputNode.cy - RECT_HEIGHT / 2,
            "x2": outputNode.cx + deltaRight,
            "y2": y + 19,
            "marker-end": "url(#markerArrow)"
        });
    }
    function addRectangle(nodeGroup, x, y, width, height, nodeSection, label, subLabel) {
        nodeGroup = nodeGroup.append("g").attr("class", nn.NodeSection[nodeSection])
            .style("opacity", 0);
        var background = nodeGroup.append("rect")
            .classed(nn.NodeSection[nodeSection], true)
            .classed("background", true);
        background.attr({
            x: x,
            y: y,
            width: width,
            height: height
        });
        var border = nodeGroup.append("rect")
            .classed(nn.NodeSection[nodeSection], true)
            .classed("border", true);
        border.attr({
            x: x,
            y: y,
            width: width,
            height: height
        });
        var text = nodeGroup.append("text").text(label);
        text.attr({
            x: RECT_WIDTH / 2,
            y: y + 12,
            "text-anchor": "middle"
        });
        text.append("tspan").text(subLabel).attr("baseline-shift", "sub");
    }
    draw.addRectangle = addRectangle;
    function drawNode(node, container, isInput, isOutput) {
        if (isInput === void 0) { isInput = false; }
        if (isOutput === void 0) { isOutput = false; }
        var x = node.cx - RECT_WIDTH / 2;
        var y = node.cy - RECT_HEIGHT / 2;
        var nodeGroup = container.append("g").attr({
            "class": "node",
            "id": "node" + node.id,
            "transform": "translate(" + x + "," + y + ")"
        });
        nodeGroup.append("rect").attr({
            x: 0,
            y: 0,
            width: RECT_WIDTH,
            height: RECT_HEIGHT,
            class: "main"
        }).style({
            fill: "none",
            "stroke-width": "1px",
            stroke: "#aaa",
            opacity: 0
        });
        nodeGroup.append("text").text(node.id)
            .attr({
            class: "main-label",
            x: RECT_WIDTH / 2,
            y: RECT_HEIGHT / 2 + 9,
            "text-anchor": "middle"
        }).style("opacity", 0);
        addRectangle(nodeGroup, 0, 0, RECT_WIDTH, RECT_HEIGHT / 4, nn.NodeSection.Output, "y", isOutput ? "output" : node.id);
        addRectangle(nodeGroup, 0, RECT_HEIGHT * 3 / 4, RECT_WIDTH, RECT_HEIGHT / 4, nn.NodeSection.Input, "x", isInput ? "input" : node.id);
        if (isInput) {
            return;
        }
        addRectangle(nodeGroup, 0, RECT_HEIGHT / 4, RECT_WIDTH, RECT_HEIGHT / 4, nn.NodeSection.DerOutput, "dE/dy", node.id);
        addRectangle(nodeGroup, 0, RECT_HEIGHT / 2, RECT_WIDTH, RECT_HEIGHT / 4, nn.NodeSection.DerInput, "dE/dx", node.id);
        var act = nodeGroup.append("g").attr("class", "activation");
        act.append("line").attr({
            x1: RECT_WIDTH / 2,
            y1: RECT_HEIGHT * 3 / 4,
            x2: RECT_WIDTH / 2,
            y2: RECT_HEIGHT / 4 + 7,
            "marker-end": "url(#markerArrow)"
        }).style({
            "stroke": "black"
        });
        act.append("circle")
            .attr({
            class: "activation",
            cx: RECT_WIDTH / 2,
            cy: RECT_HEIGHT / 2 + 5,
            r: 7
        });
        act.append("text").text("f").attr({
            x: RECT_WIDTH / 2,
            y: RECT_HEIGHT / 2 + 9,
            "text-anchor": "middle"
        });
        act.style("opacity", 0);
    }
    draw.drawNode = drawNode;
    function showActivation(nodeGroup, undo) {
        var duration = 150;
        nodeGroup.select("g.DerInput")
            .transition()
            .duration(duration)
            .style("opacity", undo ? 1 : 0);
        nodeGroup.select("g.DerOutput")
            .transition()
            .duration(duration)
            .style("opacity", undo ? 1 : 0);
        nodeGroup.select("g.activation")
            .transition()
            .duration(duration)
            .style("opacity", undo ? 0 : 1);
    }
    function updateDerWeight(rect, stroke, fill) {
        var styles = {};
        styles["fill"] = fill ? "hsl(264, 100%, 70%)" : null;
        highlightRect(rect, !stroke, true);
        rect
            .transition()
            .duration(150)
            .style(styles);
    }
    function setLoader(show) {
        d3.select("#loader").style("display", show ? "block" : "none");
    }
    function playEvent(event, undo, cleanup) {
        if (event.updatedPart.section === nn.NodeSection.DerWeight) {
            var rect = d3.selectAll("#weight" + event.updatedPart.id).select("rect.DerWeight");
            updateDerWeight(rect, !undo && !cleanup, !undo);
        }
        else {
            var rects = d3.select("#node" + event.updatedPart.id)
                .selectAll("rect." + nn.NodeSection[event.updatedPart.section]);
            highlightRect(rects.filter(".border"), undo || cleanup, true);
            updateNode(rects.filter(".background"), undo);
        }
        for (var _i = 0, _a = event.involvedParts; _i < _a.length; _i++) {
            var part = _a[_i];
            var rects = d3.select("#node" + part.id)
                .selectAll("rect." + nn.NodeSection[part.section]);
            highlightRect(rects.filter(".border"), undo || cleanup, false);
            var sourceId = Math.min(+event.updatedPart.id, +part.id);
            var destId = Math.max(+event.updatedPart.id, +part.id);
            var link = d3.select("#link" + sourceId + "-" + destId);
            var textBox = d3.select("#weight" + sourceId + "-" + destId);
            highlightLink(link, textBox, undo || cleanup);
        }
    }
    function highlightLink(line, textbox, undo) {
        highlightRect(textbox.select("rect"), undo, false);
    }
    function updateNode(rect, undo) {
        var bckNormal = {
            "opacity": 0.1
        };
        var bckUpdated = {
            "opacity": 0.7
        };
        var transition = rect.transition().duration(150);
        transition.style(undo ? bckNormal : bckUpdated);
    }
    function highlightRect(rect, undo, result) {
        rect.classed(result ? "pulsate-thick" : "pulsate-thin", !undo);
    }
    var lastSection = 0;
    var lastProgress = 0;
    var lastSubSection = -1;
    var sigmoidActivation = {
        output: function (x) {
            return 1 / (1 + Math.exp(-x));
        },
        der: function (x) {
            var output = 1 / (1 + Math.exp(-x));
            return output * (1 - output);
        }
    };
    var squareError = {
        error: function (output, target) { return 0.5 * Math.pow(output - target, 2); },
        der: function (output, target) { return output - target; }
    };
    var network = nn.buildAndDrawNetwork([1, 2, 2, 1], sigmoidActivation);
    drawErrorAndTarget(network);
    var trainingExamples = [];
    var numBadExamples = 0;
    var threshold = 0.0;
    for (var i = 0; i < NUM_EXAMPLES; i++) {
        var x = Math.random() - 0.5;
        var noise = (Math.random() - 0.5) * TARGET_NOISE;
        var target = (x + noise > threshold ? 1 : 0);
        trainingExamples.push({
            inputs: [x],
            target: target
        });
        if (x > threshold && target < 0.001 || x < threshold && target > 0.999) {
            numBadExamples++;
        }
    }
    var fracBadExamples = (numBadExamples * 100 / trainingExamples.length);
    console.log("% of bad examples: " + fracBadExamples.toFixed(2) + "%");
    var _a = nn.forwardProp(network, trainingExamples[0].inputs), fpEvents = _a[1];
    var bpEvents = nn.backProp(network, trainingExamples[0].target, squareError);
    bpEvents = bpEvents.slice(0, -1);
    function showBasicNetwork(undo, subSection, cleanup) {
        if (cleanup) {
            d3.selectAll("text.main-label").transition().style("opacity", 0);
            return;
        }
        var opacity = undo ? 0 : 1;
        d3.selectAll("text.main-label").transition().style("opacity", opacity);
        d3.selectAll("rect.main").transition().style("opacity", opacity);
        d3.selectAll("line").transition().style("opacity", opacity);
        d3.selectAll("g.textbox.weight").transition().style("opacity", opacity);
    }
    function showXYandFX(undo, subSection, cleanup) {
        if (cleanup) {
            return;
        }
        var opacity = undo ? 0 : 1;
        d3.selectAll(".node g.Input").transition().style("opacity", opacity);
        d3.selectAll(".node g.Output").transition().style("opacity", opacity);
        d3.selectAll(".node g.activation").transition().style("opacity", opacity);
    }
    function showErrorAndTarget(undo, subSection, cleanup) {
        var gError = d3.select("g.error");
        if (cleanup || undo) {
            gError.select("rect").style("stroke-width", null);
            gError.select("rect").style("stroke", null);
            gError.select("circle").style("stroke-width", null);
        }
        else {
            gError.select("rect").style("stroke-width", "2px");
            gError.select("rect").style("stroke", "black");
            gError.select("circle").style("stroke-width", "2px");
        }
        var opacity = undo ? 0 : 1;
        gError.transition().style("opacity", opacity);
    }
    function showdEdw(undo, subSection, cleanup) {
        if (cleanup || undo) {
            d3.selectAll("g.textbox.DerWeight").select("rect").style({
                "stroke": null,
                "stroke-width": null
            });
        }
        else {
            d3.selectAll("g.textbox.DerWeight").select("rect").style({
                "stroke": "black",
                "stroke-width": 2
            });
        }
        var opacity = undo ? 0 : 1;
        d3.selectAll("g.textbox.DerWeight").transition().style("opacity", opacity);
    }
    function showWholeNetwork(undo, subSection, cleanup) {
        var derInputs = d3.selectAll(".node g.DerInput");
        var derOutputs = d3.selectAll(".node g.DerOutput");
        if (cleanup || undo) {
            derInputs.select("rect.border").style({
                "stroke": null,
                "stroke-width": null
            });
            derOutputs.select("rect.border").style({
                "stroke": null,
                "stroke-width": null
            });
        }
        else {
            derInputs.select("rect.border").style({
                "stroke": "black",
                "stroke-width": 2
            });
            derOutputs.select("rect.border").style({
                "stroke": "black",
                "stroke-width": 2
            });
        }
        var opacity = undo ? 0 : 1;
        derInputs.transition().style("opacity", opacity);
        derOutputs.transition().style("opacity", opacity);
        d3.selectAll(".node g.activation").transition().style("opacity", undo ? 1 : 0);
    }
    function updateInputExample(undo, subSection, cleanup) {
        fpEvents[0].updatedPart.section = nn.NodeSection.Input;
        playEvent(fpEvents[0], undo, cleanup);
        fpEvents[0].updatedPart.section = nn.NodeSection.Output;
        playEvent(fpEvents[0], undo, cleanup);
        highlightRect(d3.select("g.error rect"), undo || cleanup, true);
        var fill = undo ? "rgba(219, 68, 55, 0.1)" : "rgba(219, 68, 55, 0.7)";
        d3.select("g.error").select("rect").transition().style("fill", fill);
    }
    function forwardPropFirstLayerInput(undo, subSection, cleanup) {
        playEvent(fpEvents[1 + subSection], undo, cleanup);
    }
    function forwardPropFirstLayerOutput(undo, subSection, cleanup) {
        playEvent(fpEvents[3 + subSection], undo, cleanup);
    }
    function forwardPropAllLayers(undo, subSection, cleanup) {
        playEvent(fpEvents[5 + subSection], undo, cleanup);
    }
    function backPropLastLayerOutput(undo, subSection, cleanup) {
        playEvent(bpEvents[0], undo, cleanup);
        highlightRect(d3.select("g.error rect"), undo || cleanup, false);
    }
    function backPropLastLayerInput(undo, subSection, cleanup) {
        playEvent(bpEvents[1], undo, cleanup);
    }
    function backPropLastLayerWeights(undo, subSection, cleanup) {
        playEvent(bpEvents[2 + subSection], undo, cleanup);
    }
    function backPropMidLayer(undo, subSection, cleanup) {
        playEvent(bpEvents[4 + subSection], undo, cleanup);
    }
    function backPropAllLayers(undo, subSection, cleanup) {
        playEvent(bpEvents[6 + subSection], undo, cleanup);
    }
    var visFunctions = [
        null,
        { method: showBasicNetwork, numSections: 1 },
        { method: showXYandFX, numSections: 1 },
        { method: showErrorAndTarget, numSections: 1 },
        { method: updateInputExample, numSections: 1 },
        { method: forwardPropFirstLayerInput, numSections: 2 },
        { method: forwardPropFirstLayerOutput, numSections: 2 },
        { method: forwardPropAllLayers, numSections: fpEvents.slice(5).length },
        { method: showdEdw, numSections: 1 },
        { method: showWholeNetwork, numSections: 1 },
        { method: backPropLastLayerOutput, numSections: 1 },
        { method: backPropLastLayerInput, numSections: 1 },
        { method: backPropLastLayerWeights, numSections: 2 },
        { method: backPropMidLayer, numSections: 2 },
        { method: backPropAllLayers, numSections: bpEvents.slice(6).length }
    ];
    function interpolate(section, subSection, newSection, newSubSection) {
        var next = newSection > section ||
            (newSection === section && newSubSection >= subSection);
        var result = [];
        var currSection = section;
        var currSubSection = subSection;
        while (true) {
            _a = next ?
                goNext(currSection, currSubSection) : goPrev(currSection, currSubSection), currSection = _a[0], currSubSection = _a[1];
            result.push([currSection, currSubSection]);
            if (currSection === section && currSubSection === subSection ||
                currSection === newSection && currSubSection === newSubSection) {
                break;
            }
        }
        return result;
        var _a;
    }
    function goNext(section, subSection) {
        var nextSection = section;
        var nextSubSection = subSection + 1;
        var numSections = visFunctions[section] != null ?
            visFunctions[section].numSections : 1;
        if (nextSubSection >= numSections) {
            nextSubSection = 0;
            nextSection = section + 1;
        }
        return [nextSection, nextSubSection];
    }
    function goPrev(section, subSection) {
        var prevSection = section;
        var prevSubSection = subSection - 1;
        if (prevSubSection < 0) {
            prevSection = section - 1;
            var prevNumSections = visFunctions[prevSection] != null ?
                visFunctions[prevSection].numSections : 1;
            prevSubSection = prevNumSections - 1;
        }
        return [prevSection, prevSubSection];
    }
    function initVis() {
        var sections = d3.selectAll("section");
        var prevNumSec = 1;
        sections.each(function () {
            d3.select(this).style({
                "margin-top": window.innerHeight / 2 * prevNumSec + "px",
                "display": "block"
            });
            prevNumSec = this.dataset.numSec || 1;
        });
        var scroll = scroller()
            .container(d3.select("#sections"));
        scroll(sections);
        scroll.on("progress", function (section, progress) {
            sections.style("opacity", function (d, i) { return i === section ? 1 : 0.1; });
            var numSections = visFunctions[section] != null ?
                visFunctions[section].numSections :
                1;
            var subSection = visFunctions[section] != null ?
                d3.bisect(d3.range(0, 1, 1 / numSections), progress) - 1 :
                0;
            if (section === lastSection && subSection === lastSubSection) {
                return;
            }
            interpolate(lastSection, lastSubSection, section, subSection).forEach(function (next) {
                var currSection = next[0], currSubSection = next[1];
                var _a = goNext(currSection, currSubSection), nextSection = _a[0], nextSubSection = _a[1];
                var _b = goPrev(currSection, currSubSection), prevSection = _b[0], prevSubSection = _b[1];
                if (visFunctions[prevSection] != null) {
                    console.log("cleanup", prevSection, prevSubSection);
                    visFunctions[prevSection].method(false, prevSubSection, true);
                }
                if (visFunctions[nextSection] != null) {
                    console.log("undo", nextSection, nextSubSection);
                    visFunctions[nextSection].method(true, nextSubSection, false);
                }
                if (visFunctions[currSection] != null) {
                    console.log("do", currSection, currSubSection);
                    visFunctions[currSection].method(false, currSubSection, false);
                }
            });
            lastSection = section;
            lastSubSection = subSection;
            lastProgress = progress;
        });
        document.body.scrollTop = document.documentElement.scrollTop = 0;
    }
    setTimeout(initVis, 500);
})(draw || (draw = {}));
//# sourceMappingURL=draw.js.map