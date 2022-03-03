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

const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;
const MarkdownBlock = CompLibrary.MarkdownBlock;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);
const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;

class TutorialHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">
                Welcome to the PyTorch3D Tutorials
              </h1>
            </header>
            <p>
              Here you can learn about the structure and applications of
              PyTorch3D from examples which are in the form of ipython
              notebooks.
            </p>
            <h3> Run interactively </h3>
            <p>
              At the top of each example you can find a button named{' '}
              <strong>"Run in Google Colab"</strong> which will open the
              notebook in{' '}
              <a href="https://colab.research.google.com/notebooks/intro.ipynb">
                {' '}
                Google Colaboratory{' '}
              </a>{' '}
              where you can run the code directly in the browser with access to
              GPU support - it looks like this:
            </p>
            <div className="tutorialButtonsWrapper">
              <div className="tutorialButtonWrapper buttonWrapper">
                <a className="tutorialButton button" target="_blank">
                  <img
                    className="colabButton"
                    align="left"
                    src="/img/colab_icon.png"
                  />
                  {'Run in Google Colab'}
                </a>
              </div>
            </div>
            <p>
              {' '}
              You can modify the code and experiment with varying different
              settings. Remember to install the latest stable version of
              PyTorch3D and its dependencies. Code to do this with pip is
              provided in each notebook.{' '}
            </p>
            <h3> Run locally </h3>
            <p>
              {' '}
              There is also a button to download the notebook and source code to
              run it locally.{' '}
            </p>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
