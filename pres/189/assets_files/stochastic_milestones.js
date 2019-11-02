Math.randn = function() {
  var x1, x2, rad, y1;
  do {
    x1 = 2 * this.random() - 1;
    x2 = 2 * this.random() - 1;
    rad = x1 * x1 + x2 * x2;
  } while(rad >= 1 || rad == 0);
  var c = this.sqrt(-2 * Math.log(rad) / rad);
  return x1 * c;
};

function renderStochasticMilestones(div, updateTick) {

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

  var lam = 1
  var epsilon = 3

  var f = function(x) { 
    var fx = 0.5*(lam*x[0]*x[0])
    var g  = [lam*x[0]]
    return [fx, g]
  }


  var stochasticf = function(x) { 
    var fx = 0.5*(lam*x[0]*x[0])
    var g  = [lam*x[0] + epsilon*Math.randn()]
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

    var Rmat = function (i) {
      return [[          beta,           lam ],
              [ -1*alpha*beta, 1 - alpha*lam ]]
    }

    var R = Rmat(1)
    var w0 = [Math.sqrt(3)]
    var v = runMomentum(f, w0, alpha, beta, totalIters)
    var fxstack = []
    var errx = [epsilon, -alpha*epsilon]
    var errsum = 0

    var nsamples = 1
    svsamples = []
    for (var i = 0; i < nsamples; i++) {
      var sv = runMomentum(stochasticf, w0, alpha, beta, totalIters)
      svsamples.push(sv[1])
    }
    fxaverage = []
    for (var i = 0; i < v[0].length; i++) {
      var mean = 0
      for (var j = 0; j < nsamples; j++) {
        mean = mean + (lam*svsamples[j][i][0]*svsamples[j][i][0])/2
      }
      fxaverage.push([mean/nsamples])
    }
    var stochasticfx = []
    // Add contributions to the objective
    for (var i = 0; i < v[0].length; i++) {
      errsum = errsum + errx[1]*errx[1]/2
      errx = numeric.dot(R, errx)
      var x = v[1][i]
      fxstack.push([errsum, lam*x[0]*x[0]/2])
      stochasticfx.push([lam*sv[1][i][0]*sv[1][i][0]/2])
    }
    v.push(fxstack)

    return {deterministic:v, stochastic:fxaverage}
  }


  var stackedBar = stackedBarchartGen(totalIters, 2)
                    .col(colorbrewer.BuPu)
                    .translatey(30)
                    .translatex(110)
                    .highlightcol("darkblue")(div) 


  var stackedBar2 = stackedBarchartGen(totalIters, 1)
                    .col(colorbrewer.BuPu)
                    .translatey(30)
                    .translatex(110)
                    .cr(1.2)
                    .copacity(0.5)
                    .lineopacity(0)
                    .drawgrid(false)
                    .highlightcol("darkblue")(div) 
    

  div.append("rect")
     .attr("x", 0)
     .attr("y", 0)
     .attr("width", 1000)
     .attr("height", 30)
     .attr("fill", "white")

  var seperation = 14


  var progressmeter = div.append("g")

  var textl = progressmeter.append("g")

  textl.append("line")
       .attr("y1",-5)
       .attr("y2",-5)
       .attr("x1",110)
       .attr("x2",120)
       .attr("stroke","black")
       .attr("stroke-width", 1.5)
       .attr("marker-end", "url(#arrowhead)")

  textl.append("text")
       .attr("class", "figtext2")
       .text("Fine-tuning phase")

  var textr = progressmeter.append("g")

  textr.append("text")
       .attr("class", "figtext2")
       .attr("text-anchor", "end")
       .text("Transient phase")


  textr.append("line")
       .attr("y1",-5)
       .attr("y2",-5)
       .attr("x1",-100)
       .attr("x2",-110)
       .attr("stroke","black")
       .attr("stroke-width", 1.5)
       .attr("marker-end", "url(#arrowhead)")

  var divider2 = progressmeter.append("line")
            .style("stroke", "white")
            .style("stroke-width",5)
            .attr("opacity", 0.9)

  var divider = progressmeter.append("line")
            .style("stroke", "black")
            .style("stroke-width",1.5)
            .attr("opacity", 0.9)

  var updateStep = function(alpha, beta) {
    var tracebath =  getTrace(alpha/lambda[2], beta)
    var trace = tracebath.deterministic
    var strace = tracebath.stochastic

    // Update the milestones on the slider
    var milestones = [0,0]
    for (var i = 0; i < trace[1].length; i++) {
      if (trace[2][i][1] > trace[2][i][0]) { milestones[0] = i;  milestones[1] = i; } else { break}
    }
    stackedBar.update(trace[2], milestones)
    stackedBar2.update(strace, milestones)
    var endpoint = stackedBar.stack[0].selectAll("line").nodes()[milestones[0]]
    var stack = endpoint.getBBox()
    var ctm = endpoint.getCTM()

    if (milestones[0] < 150) {
      textl.attr("transform", "translate(" + (stack.x + 10) + ",-15)").style("visibility", "visible")
      textr.attr("transform", "translate(" + (stack.x - 10) + ",-16)").style("visibility", "visible")
      divider.attr("x2", stack.x)
              .attr("y2", -25)
              .attr("x1", stack.x)
              .attr("y1", 160)      
              .style("visibility", "visible")
      divider2.attr("x2", stack.x)
              .attr("y2", -25)
              .attr("x1", stack.x)
              .attr("y1", 160)      
              .style("visibility", "visible")              
    } else {
      endpointend = stackedBar.stack[0].selectAll("line").nodes()[149].getBBox().x
      console.log(endpointend)
      textr.attr("transform", "translate(" + (endpointend + 10) + ",-16)").style("visibility", "visible")      
      divider.style("visibility", "hidden")
      textl.style("visibility", "hidden")
      divider2.style("visibility", "hidden")
    }
    setTM(progressmeter.node(), ctm)

  }

  updateStep(100*2/(101.5), 0)

  return updateStep
}