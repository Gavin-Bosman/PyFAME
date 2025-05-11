---
layout: doc
title: Contributing
prev:
    text: 'Authors'
    link: '/about/authors'
next: 
    text: 'Changelog'
    link: '/about/changelog'
---

# Contributing

For developers who are looking to contribute to the project, PyFAME can be cloned in typical SSH fashion:

``` sh
git clone git@github.com:Gavin-Bosman/PyFAME.git
```

Once you have the repository cloned locally, you will need to install dependencies via

``` sh
pip install -r requirements.txt
```

Additionally, if you would like to be able to contribute to the packages test suite or documentation site, you will need to install additional dev dependencies:

::: info
This documentation site is built with vitepress, which requires you to have both Node.js>=20.0.0 and npm>=10.0.0 installed on your system. For more information on installing and setting up Node, [see here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
:::

``` sh
    pip install -r requirements_dev.txt
    npm install
```