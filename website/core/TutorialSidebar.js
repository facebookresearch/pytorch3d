/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');
const fs = require('fs-extra');
const path = require('path');
const join = path.join;
const CWD = process.cwd();

const CompLibrary = require(join(
  CWD,
  '/node_modules/docusaurus/lib/core/CompLibrary.js',
));
const SideNav = require(join(
  CWD,
  '/node_modules/docusaurus/lib/core/nav/SideNav.js',
));

const Container = CompLibrary.Container;

const OVERVIEW_ID = 'tutorial_overview';

class TutorialSidebar extends React.Component {
  render() {
    const {currentTutorialID} = this.props;
    const current = {
      id: currentTutorialID || OVERVIEW_ID,
    };

    const toc = [
      {
        type: 'CATEGORY',
        title: 'Tutorials',
        children: [
          {
            type: 'LINK',
            item: {
              permalink: 'tutorials/',
              id: OVERVIEW_ID,
              title: 'Overview',
            },
          },
        ],
      },
    ];

    const jsonFile = join(CWD, 'tutorials.json');
    const normJsonFile = path.normalize(jsonFile);
    const json = JSON.parse(fs.readFileSync(normJsonFile, {encoding: 'utf8'}));

    Object.keys(json).forEach(category => {
      const categoryItems = json[category];
      const items = [];
      categoryItems.map(item => {
        items.push({
          type: 'LINK',
          item: {
            permalink: `tutorials/${item.id}`,
            id: item.id,
            title: item.title,
          },
        });
      });

      toc.push({
        type: 'CATEGORY',
        title: category,
        children: items,
      });
    });

    return (
      <Container className="docsNavContainer" id="docsNav" wrapper={false}>
        <SideNav
          language={'tutorials'}
          root={'tutorials'}
          title="Tutorials"
          contents={toc}
          current={current}
        />
      </Container>
    );
  }
}

module.exports = TutorialSidebar;
