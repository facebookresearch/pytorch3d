This website was created with [Docusaurus](https://docusaurus.io/).

# Building the PyTorch3D website

## Install 

1. Make sure all the dependencies for the website are installed:

```sh
# Install dependencies
$ yarn

or 

$ npm install docusaurus-init
```

2. Run your dev server:

```sh
# Start the site
$ yarn start

or
$ ./node_modules/docusaurus/lib/start-server.js
```

## Build the tutorials

We convert the ipython notebooks to html using `parse_tutorials.py` which is found in the scripts folder at the root of the PyTorch3D directory.

Before running this script install the following dependencies:

```
pip install nbformat==4.4.0 nbconvert==5.3.1 ipywidgets==7.5.1 tornado==4.2 bs4
```

Install yarn:

```
brew install yarn

# or 

curl -o- -L https://yarnpkg.com/install.sh | bash
```

Then run the build script:

```
bash scripts/build_website.sh
```

This will build the docusaurus website and run a script to parse the tutorials and generate:
- `.html` files in the `website/_tutorials` folder
- `.js` files in the `website/pages/tutorials` folder
- `.py`/`.ipynb` files in the `website/static/files` folder


TODO: Add support for latex in markdown in jupyter notebooks and embedded images. 

## Build and publish the website

The following script will build the tutorials and the website and push to the gh-pages 
branch of `github.com/facebookresearch/pytorch3d`.

```
bash scripts/publish_website.sh
```


## Add a new tutorial

The tutorials to include in the website are listed in `website/tutorials.json`. If you create a new tutorial add an entry to the list in this file. This is needed in order to generate the sidebar for the tutorials page. 


## Edit the landing page

To change the content of the landing page modify: `website/pages/en/index.js`. 


## Edit the tutorials page

To change the content of the tutorials home page modify: `website/pages/tutorials/index.js`. 


---------------------------------------------------------

## Docusaurus docs

- [Get Started in 5 Minutes](#get-started-in-5-minutes)
- [Directory Structure](#directory-structure)
- [Editing Content](#editing-content)
- [Adding Content](#adding-content)
- [Full Documentation](#full-documentation)


## Directory Structure

Your project file structure should look something like this

```
my-docusaurus/
  docs/
    doc-1.md
    doc-2.md
    doc-3.md
  website/
    blog/
      2016-3-11-oldest-post.md
      2017-10-24-newest-post.md
    core/
    node_modules/
    pages/
    static/
      css/
      img/
    package.json
    sidebars.json
    siteConfig.js
```

# Editing Content

## Editing an existing docs page

Edit docs by navigating to `docs/` and editing the corresponding document:

`docs/doc-to-be-edited.md`

```markdown
---
id: page-needs-edit
title: This Doc Needs To Be Edited
---

Edit me...
```

For more information about docs, click [here](https://docusaurus.io/docs/en/navigation)

## Editing an existing blog post

Edit blog posts by navigating to `website/blog` and editing the corresponding post:

`website/blog/post-to-be-edited.md`

```markdown
---
id: post-needs-edit
title: This Blog Post Needs To Be Edited
---

Edit me...
```

For more information about blog posts, click [here](https://docusaurus.io/docs/en/adding-blog)

# Adding Content

## Adding a new docs page to an existing sidebar

1. Create the doc as a new markdown file in `/docs`, example `docs/newly-created-doc.md`:

```md
---
id: newly-created-doc
title: This Doc Needs To Be Edited
---

My new content here..
```

1. Refer to that doc's ID in an existing sidebar in `website/sidebars.json`:

```javascript
// Add newly-created-doc to the Getting Started category of docs
{
  "docs": {
    "Getting Started": [
      "quick-start",
      "newly-created-doc" // new doc here
    ],
    ...
  },
  ...
}
```

For more information about adding new docs, click [here](https://docusaurus.io/docs/en/navigation)

## Adding a new blog post

1. Make sure there is a header link to your blog in `website/siteConfig.js`:

`website/siteConfig.js`

```javascript
headerLinks: [
    ...
    { blog: true, label: 'Blog' },
    ...
]
```

2. Create the blog post with the format `YYYY-MM-DD-My-Blog-Post-Title.md` in `website/blog`:

`website/blog/2018-05-21-New-Blog-Post.md`

```markdown
---
author: Frank Li
authorURL: https://twitter.com/foobarbaz
authorFBID: 503283835
title: New Blog Post
---

Lorem Ipsum...
```

For more information about blog posts, click [here](https://docusaurus.io/docs/en/adding-blog)

## Adding items to your site's top navigation bar

1. Add links to docs, custom pages or external links by editing the headerLinks field of `website/siteConfig.js`:

`website/siteConfig.js`

```javascript
{
  headerLinks: [
    ...
    /* you can add docs */
    { doc: 'my-examples', label: 'Examples' },
    /* you can add custom pages */
    { page: 'help', label: 'Help' },
    /* you can add external links */
    { href: 'https://github.com/facebook/docusaurus', label: 'GitHub' },
    ...
  ],
  ...
}
```

For more information about the navigation bar, click [here](https://docusaurus.io/docs/en/navigation)

## Adding custom pages

1. Docusaurus uses React components to build pages. The components are saved as .js files in `website/pages/en`:
1. If you want your page to show up in your navigation header, you will need to update `website/siteConfig.js` to add to the `headerLinks` element:

`website/siteConfig.js`

```javascript
{
  headerLinks: [
    ...
    { page: 'my-new-custom-page', label: 'My New Custom Page' },
    ...
  ],
  ...
}
```

For more information about custom pages, click [here](https://docusaurus.io/docs/en/custom-pages).

# Full Documentation

Full documentation can be found on the [website](https://docusaurus.io/).
