/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * @format
 **/

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
              Pytorch3D from examples which are in the form of ipython
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
              settings. Remember to install pytorch, torchvision, fvcore and
              pytorch3d in the first cell of the colab notebook by running:{' '}
            </p>
            <MarkdownBlock>{bash`!pip install torch torchvision
!pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'`}</MarkdownBlock>
            This installs the latest stable version of PyTorch3D from github.
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
