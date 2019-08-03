function renderMilestones(div, updateTick) {
  var lambda = [1,10,100]
  var totalIters = 151  

  div.append("defs").append("marker")
      .attr("id", "arrowhead")
      .attr("refX", 3) 
      .attr("refY", 2)
      .attr("markerWidth", 4)
      .attr("markerHeight", 4)
      .attr("orient", "auto")
      .append("path")
          .attr("d", "M 0,0 V 4 L4,2 Z"); //this is actual shape for arrowhead

  var markers = []
  for (var i = 0; i < 3; i ++) {
    var marker = div.append("defs").append("marker")
      .attr("id", "arrowhead" + i)
      .attr("refX", 0) 
      .attr("refY", 1)
      .attr("markerWidth", 4)
      .attr("markerHeight", 4)
      .attr("orient", "auto")
      .append("path")
        .attr("d", "M 0,0 V 2 L2,1 Z") //this is actual shape for arrowhead
        .attr("fill", colorbrewer.RdPu[3][i])
      markers.push(marker)
  }

  var f = function(x) { 
    var fx = 0.5*(lambda[0]*x[0]*x[0] + lambda[1]*x[1]*x[1] + lambda[2]*x[2]*x[2])
    var g  = [lambda[0]*x[0], lambda[1]*x[1], lambda[2]*x[2]]
    return [fx, g]
  }

  /*
    Gets information about iterates with certain parameter alpha and beta.
    > v = getTrace(0.5, 0)
    v[0] -> z iterates
    v[1] -> w iterates
    v[2] -> [c1, c2, c3] where c1, c2, c3 are the contributions to the loss
  */
  function getTrace(alpha, beta) {
    var w0 = [1,1/Math.sqrt(lambda[1]),1/Math.sqrt(lambda[2])]
    var v = runMomentum(f, w0, alpha, beta, totalIters)
    var fxstack = []
    // Add contributions to the objective
    for (var i = 0; i < v[0].length; i++) {
      var x = v[1][i]
      fxstack.push([lambda[0]*x[0]*x[0]/2, lambda[1]*x[1]*x[1]/2, lambda[2]*x[2]*x[2]/2 ])
    }
    v.push(fxstack)
    return v
  }

  var stackedBar = stackedBarchartGen(totalIters, 3)(div) 
  
  var seperation = 14

  var r = []
  var lines = []

  var progressmeter = div.append("g")
  for (var i = 0; i < 3; i ++) {
    var ri = progressmeter.append("line")
      .attr("x1", stackedBar.X(-1) + "px")
      .attr("y1", (202 + i*seperation)+ "px")
      .attr("stroke", colorbrewer.RdPu[3][i])
      .attr("y2", (202 + i*seperation) + "px")
      .attr("stroke-width", 4)
    r.push(ri)

    var linei = progressmeter.append("line")
              .style("stroke", "black")
              .style("stroke-width",1.5)
              .attr("marker-end", "url(#arrowhead)")
              .attr("opacity", 0.6)
    lines.push(linei)

    progressmeter.append("text")
    .attr("class", "figtext2")
    .attr("x", 0)
    .attr("y", 206 + i*seperation)
    .attr("text-anchor", "end")
    .attr("fill", "gray")    
    .html((i == 0) ? "Eigenvalue " + (i+1) : (i+1) )
  }


  var updateStep = function(alpha, beta) {
    var trace = getTrace(alpha/lambda[2], beta)
    // Update the milestones on the slider
    var milestones = [0,0,0]
    for (var i = 0; i < trace[1].length; i++) {
      if (trace[2][i][0] > 0.01) { milestones[2] = i }
      if (trace[2][i][1] > 0.01) { milestones[1] = i }
      if (trace[2][i][2] > 0.01) { milestones[0] = i }
    }
    stackedBar.update(trace[2], milestones)

    for (var i = 0; i < 3; i++) {

      var endpoint = stackedBar.stack[i].selectAll("line").nodes()[milestones[2-i]]
      var stack = endpoint.getBBox()
      var ctm = endpoint.getCTM()
      if (milestones[2-i] < 150) {
        lines[i].attr("x2", stack.x)
                .attr("y2", stack.y + 5)
                .attr("x1", stack.x)
                .attr("y1", 203.5 + seperation*(i))      
                .style("visibility", "visible")
        r[i].attr("marker-end", "url()")   
      } else {
        lines[i].style("visibility", "hidden")
        r[i].attr("marker-end", "url(#arrowhead" + i + ")")   
      }
      //setTM(lines[i].node(), ctm) // transform the lines into stackedplot space
      r[i].attr("x2", (stackedBar.X(milestones[2-i]) - 2) + "px")
      
      //setTM(r[i].node(), ctm) // transform the lines into stackedplot space
      setTM(progressmeter.node(), ctm)
    }
  }

  updateStep(100*2/(101.5), 0)

  return updateStep
}