(function () {
    function onReady(fn) {
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
            fn();
        } else {
            document.addEventListener('DOMContentLoaded', fn, { once: true });
        }
    }

    onReady(function () {
        var canvas = document.getElementById('bg-particles');
        if (!canvas || !canvas.getContext) {
            return;
        }

        var ctx = canvas.getContext('2d');
        var dpr = window.devicePixelRatio || 1;
        var width = 0;
        var height = 0;
        var particles = [];
        var animationId = null;
        var BASE_SPEED = 0.04;

        function randomBetween(min, max) {
            return Math.random() * (max - min) + min;
        }

        function setCanvasSize() {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width * dpr;
            canvas.height = height * dpr;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);
        }

        function createParticles() {
            var targetCount = Math.max(48, Math.min(140, Math.floor(width / 12)));
            particles.length = 0;
            for (var i = 0; i < targetCount; i++) {
                particles.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    radius: randomBetween(1.6, 4.5),
                    velocityX: randomBetween(-BASE_SPEED, BASE_SPEED) * randomBetween(0.4, 1.6),
                    velocityY: randomBetween(-BASE_SPEED, BASE_SPEED) * randomBetween(0.4, 1.6),
                    opacity: randomBetween(0.25, 0.55)
                });
            }
        }

        function updateParticle(p) {
            p.x += p.velocityX;
            p.y += p.velocityY;

            if (p.x < -10) {
                p.x = width + 10;
            } else if (p.x > width + 10) {
                p.x = -10;
            }

            if (p.y < -10) {
                p.y = height + 10;
            } else if (p.y > height + 10) {
                p.y = -10;
            }
        }

        function render() {
            ctx.clearRect(0, 0, width, height);

            for (var i = 0; i < particles.length; i++) {
                var p = particles[i];
                updateParticle(p);
                ctx.beginPath();
                ctx.fillStyle = 'rgba(216, 233, 255,' + p.opacity.toFixed(3) + ')';
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fill();
            }

            animationId = window.requestAnimationFrame(render);
        }

        function start() {
            if (animationId !== null) {
                return;
            }
            animationId = window.requestAnimationFrame(render);
        }

        function stop() {
            if (animationId !== null) {
                window.cancelAnimationFrame(animationId);
                animationId = null;
            }
        }

        function handleVisibility() {
            if (document.hidden) {
                stop();
            } else {
                start();
            }
        }

        function handleResize() {
            setCanvasSize();
            createParticles();
        }

        handleResize();
        start();

        window.addEventListener('resize', handleResize);
        document.addEventListener('visibilitychange', handleVisibility);
    });
})();
