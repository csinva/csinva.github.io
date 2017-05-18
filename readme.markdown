# Timeline Jekyll Theme

Timeline is multipurpose, single page Jekyll theme that includes a timeline section. 

**Example sites**:

- [Demo site](http://kirbyt.github.io/timeline-jekyll-theme)
- [kirbyturner.com](http://www.kirbyturner.com)

This theme is a mashup of two other themes:

- [Grayscale by Start Bootstrap](https://github.com/IronSummitMedia/startbootstrap-grayscale)
- [Agency Jekyll Theme](https://github.com/y7kim/agency-jekyll-theme)

Big thanks to the creators of Grayscale and Agency. This theme would not be possible without their hard work.

## How to Use

Timeline works in a similar fashion as a Jekyll blog site with two differences:

1. Pages are displayed as sections within the single HTML web page.
2. Posts are displayed as timeline entries.

### Creating a Section

To add a section, create a new *.html* file such as *about.html*. This will add the section to the navigation menu and display the section as part of the single page HTML. The file should include the following YAML front matter:

- **layout** It's value should always be `null`.
- **title** This is the text displayed in the navigation menu.
- **section-type** This identifies the section type, or layout for the section. The possible values are:
    + `default`
    + `contact`
    + `timeline`

Use the **section-type** `default` when you want to display the content in a regular section.

Use the **section-type** `contact` to display a contact section. A contact section can optionally contain an email address and a list of social networks that are display below the content. See *contact.html* for an example of setting the email address and social networks.

Use the **section-type** `timeline` to display a timeline of entries. 

## More About Timelines

A timeline is a list of post sorted by date. To create a new timeline entry, add a new post to the *_posts* directory. The post should have the following YAML front matter properties:

- **layout** It's value should always be `null`.
- **title** This is the header text for the career entry.
- **subtitle** (Optional) This is the sub-header text for the career entry.
- **image** (Optional) This is a reference to the thumbnail image displayed with the career entry. If blank or not present, then `site.career-img` defined in the *_config.yml* is used.

Example timeline post front matter:

```
---
layout: null
title: 2014
subtitle:
image: "img/timeline/2014.png"
---
```

# License

Code released under the [Apache 2.0][license] license.

**Portions copyrighted by**

Copyright 2013-2015 Iron Summit Media Strategies, LLC.  
Copyright 2014 Rick Kim (y7kim).  
Copyright 2015 Kirby Turner

[license]: https://github.com/kirbyt/timeline-jekyll-theme/blob/master/LICENSE
