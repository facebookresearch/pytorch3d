/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @licenselint-loose-mode

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

// List of projects/orgs using your project for the users page.
const users = [
  {
    caption: 'User1',
    // You will need to prepend the image path with your baseUrl
    // if it is not '/', like: '/test-site/img/image.jpg'.
    image: '/img/undraw_open_source.svg',
    infoLink: 'https://www.facebook.com',
    pinned: true,
  },
];

const baseUrl = '/'

const siteConfig = {
  title: 'PyTorch3D', // Title for your website.
  tagline: 'A library for deep learning with 3D data',
  url: 'https://pytorch3d.org', // Your website URL
  baseUrl: baseUrl, // Base URL for your project */
  projectName: 'pytorch3d',
  organizationName: 'facebookresearch',
  customDocsPath: 'docs/notes',
  headerLinks: [
    {doc: 'why_pytorch3d', label: 'Docs'},
    {page: 'tutorials', label: 'Tutorials'},
    {href: "https://pytorch3d.readthedocs.io/", label: 'API'},
    {href: "https://github.com/facebookresearch/pytorch3d", label: 'GitHub'},
  ],

  // If you have users set above, you add it here:
  users,

  /* path to images for header/footer */
  headerIcon: 'img/pytorch3dfavicon.png',
  footerIcon: 'img/pytorch3dfavicon.png',
  favicon: 'img/pytorch3dfavicon.png',

  /* Colors for website */
  colors: {
    primaryColor: '#812CE5',
    secondaryColor: '#FFAF00',
  },

  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Copyright \u{00A9} ${new Date().getFullYear()} Meta Platforms, Inc`,

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: 'default',
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: ['https://buttons.github.io/buttons.js'],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/pytorch3dlogoicon.svg',
  twitterImage: 'img/pytorch3dlogoicon.svg',

   // Google analytics
   gaTrackingId: 'UA-157376881-1',

  // For sites with a sizable amount of content, set collapsible to true.
  // Expand/collapse the links and subcategories under categories.
  // docsSideNavCollapsible: true,

  // Show documentation's last contributor's name.
  enableUpdateBy: true,

  // Show documentation's last update time.
  // enableUpdateTime: true,
};

module.exports = siteConfig;
