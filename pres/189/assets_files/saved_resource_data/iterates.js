/*
  Make a standard "path of descent" plot which shows iterates
  on a 2D optimization landscape

  Takes in : f, the objective function
             Name of div where the graphic is rendered
             update, which gets passed in the objective
             values at every iteration

  Returns : callback `changeParams` to change alpha, beta
*/
function genIterDiagram(f, xstar, axis) {

  /*
    Configurable Parameters
  */
  var w = 984
  var h = 300
  var totalIters = 150
  var state_beta = 0.0
  var state_alpha = 0.001
  var num_contours = 15
  var onDrag = function() {}
  var w0 =[-1.21, 0.853]
  var strokeColor = "#ff6600"
  var showOptimum = true
  var showSolution = true
  var pathWidth = 1.5
  var circleRadius = 2
  var pointerScale = 1
  var showStartingPoint = true

  function renderIterates(div) {

    // Render the other stuff
    var intDiv = div.style("width", w + "px")
      .style("height", h + "px")

    // Render Contours
    var plotCon = contour_plot.ContourPlot(w,h)
        .f(function(x,y) { return f([x,y])[0] })
        .drawAxis(false)
        .xDomain(axis[0])
        .yDomain(axis[1])
        .contourCount(num_contours)
        .minima([{x:xstar[0], y:xstar[1]}]);

    var elements = plotCon(intDiv);

    var svg = intDiv.append("div")
      .append("svg")
        .style("position", 'absolute')
        .style("left", 0)
        .style("top", 0)
        .style("width", w + "px")
        .style("height", h + "px")
        .style("z-index", 2)

    var X = d3.scaleLinear().domain(axis[0]).range([0, w])
    var Y = d3.scaleLinear().domain(axis[1]).range([0, h])

    // Rendeer Draggable dot
    var circ = svg.append("g") //use xlink:href="cafe-18.svg#svg4619">
      .attr("transform", "translate(" + X(w0[0]) + "," +  Y(w0[1]) + ")")
      .call(d3.drag().on("drag", function() {
        var pt = d3.mouse(svg.node())
        var x = X.invert(pt[0])
        var y = Y.invert(pt[1])
        this.setAttribute("transform","translate(" + pt[0] + "," +  pt[1] + ")")
        w0 = [x,y]
        onDrag(w0)
        iter(state_alpha, state_beta, w0);
      }))

    circ.append("use")
      .style("cursor", "pointer")
      .attr("xlink:href", "#pointerThingy")
      .attr("transform", "scale(" + pointerScale + ")")

    if (showStartingPoint) {
      circ.append("text")
        .style("cursor", "pointer")
        .attr("class", "figtext")
        .attr("transform", "translate(20,3)")
        .html("Starting Point")
    }

    var iterColor = d3.scaleLinear().domain([0, totalIters]).range([strokeColor, strokeColor])

    var update2D = plot2dGen(X, Y, iterColor)
                    .stroke(strokeColor)
                    .pathWidth(pathWidth)
                    .circleRadius(circleRadius)(svg)


    if (showOptimum) {
      // Append x^s var showSolution = falsetar
      var pxstar = ringPathGen(7,50,14)([X(xstar[0]), Y(xstar[1])],
                                        [X(xstar[0]), Y(xstar[1]) - 15])
      svg.append("circle").attr("cx", X(xstar[0])).attr("cy", Y(xstar[1])).attr("r", 7).attr("stroke","#3f5b75").attr("stroke-width",1).attr("fill","none")
      svg.append("path").attr("d", pxstar.d).attr("stroke","#3f5b75").attr("stroke-width",1).attr("fill","none")
      svg.append("text")
        .attr("class","figtext")
        .attr("transform", "translate(" + pxstar.label[0] + "," + (pxstar.label[1]) + ")" )
        .html("Optimum")
    }


    if (showSolution) {
      var pxsol = ringPathGen(7,43.36,14)([X(0), Y(0)],
                                      [X(0), Y(0) + 20])
      var solcirc = svg.append("circle").attr("cx", X(0)).attr("cy", Y(0)).attr("r", 7).attr("stroke",strokeColor).attr("stroke-width",1).attr("fill","none")
      var solpath = svg.append("path").attr("d", pxsol.d).attr("stroke",strokeColor).attr("stroke-width",1).attr("fill","none")
      var sollabel = svg.append("text")
                      .attr("class","figtext")
                      .attr("transform", "translate(" + pxsol.label[0] + "," + (pxsol.label[1]) + ")" )
                      .html("Solution")
    }
    function updateSol(x,y) {
      if (showSolution) {
        var pxsol = ringPathGen(7,50,14)([X(x), Y(y)], [X(x), Y(y) + 15])
        solcirc.attr("cx", X(x)).attr("cy", Y(y))
        solpath.attr("d", pxsol.d)
        sollabel.attr("transform", "translate(" + pxsol.label[0] + "," + (pxsol.label[1]) + ")" )
      }
    }

    // svg.append("rect")
    //   .attr("width", 50)
    //   .attr("height",14)
    //   .attr("x", pxstar.label[0] )
    //   .attr("y", pxstar.label[1])

    function iter(alpha, beta, w0) {

      // Update Internal state of alpha and beta
      state_alpha = alpha
      state_beta  = beta

      // Generate iterates
      var OW = runMomentum(f, w0, alpha, beta, totalIters)
      var W = OW[1]

      update2D(W)
      updateSol(OW[1][150][0], OW[1][150][1])
      circ.attr("transform", "translate(" +  X(w0[0]) + "," + Y(w0[1]) + ")" )
      circ.moveToFront()

    }

    iter(state_alpha, state_beta, w0);

    return { control:iter,
             w0:function() { return w0 },
             alpha:function() { return state_alpha },
             beta:function() {return state_beta} }

  }

  renderIterates.showStartingPoint = function(_) {
    showStartingPoint = _; return renderIterates;
  }

  renderIterates.pointerScale = function(_) {
    pointerScale = _; return renderIterates;
  }

  renderIterates.circleRadius = function(_) {
    circleRadius = _; return renderIterates;
  }

  renderIterates.pathWidth = function(_) {
    pathWidth = _; return renderIterates;
  }

  renderIterates.showSolution = function(_) {
    showSolution = _; return renderIterates;
  }

  renderIterates.showOptimum = function(_) {
    showOptimum = _; return renderIterates;
  }

  renderIterates.strokeColor = function(_) {
    strokeColor = _; return renderIterates;
  }

  renderIterates.width = function (_) {
    w = _; return renderIterates;
  }

  renderIterates.height = function (_) {
    h = _; return renderIterates;
  }

  renderIterates.iters = function (_) {
    totalIters = _; return renderIterates;
  }

  renderIterates.drag = function (_) {
    onDrag = _; return renderIterates;
  }

  renderIterates.init = function (_) {
    w0 = _; return renderIterates;
  }

  renderIterates.alpha = function (_) {
    state_alpha = _; return renderIterates;
  }

  renderIterates.beta = function (_) {
    state_beta = _; return renderIterates;
  }

  return renderIterates

}
