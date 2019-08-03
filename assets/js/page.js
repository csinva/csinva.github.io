(function (window) {
	var events = {};

	events.record = function (eventName) {
	    ga('send', {
			  'hitType': 'event',
			  'eventCategory': eventName,
			  'eventAction': 'event actioned'
			});
    };

    window.eventsWrapper = events;
}(window));

var recordThemeClick = function() {
    var attribute = this.getAttribute("data-theme-name");

		if (this.getAttribute('class') == 'download-link') {
			window.eventsWrapper.record('Downloaded ' + attribute);
		} else {
			window.eventsWrapper.record('GitHub ' + attribute);
		}
};

var download_link = document.getElementsByClassName("download-link");

for (var i=0; i<download_link.length; i++){
    download_link[i].addEventListener('click', recordThemeClick);
}

var github_link = document.getElementsByClassName("github-link");

for (var i=0; i<github_link.length; i++){
    github_link[i].addEventListener('click', recordThemeClick);
}

var recordServiceClick = function() {
    var attribute = this.getAttribute("data-service-name");

		window.eventsWrapper.record('Clicked Service ' + attribute);
};

var service_link = document.getElementsByClassName("service-link");

for (var i=0; i<service_link.length; i++){
    service_link[i].addEventListener('click', recordServiceClick);
}
