/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;
const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;
class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="splashLogo">
        <img src={props.img_src} alt="Project Logo" />
      </div>
    );

    const ProjectTitle = props => (
      <h2 className="projectTitle">
        <small>{props.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <Logo img_src={baseUrl + 'img/pytorch3dlogowhite.svg'} />
        <div className="inner">
          <ProjectTitle tagline={siteConfig.tagline} title={siteConfig.title} />
          <PromoSection>
            <Button href={docUrl('why_pytorch3d.html')}>Docs</Button>
            <Button href={`${baseUrl}tutorials/`}>Tutorials</Button>
            <Button href={'#quickstart'}>Get Started</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

function SocialBanner() {
  return (
    <div className="socialBanner">
      <div>
        Support Ukraine 🇺🇦{' '}
        <a href="https://opensource.fb.com/support-ukraine">
          Help Provide Humanitarian Aid to Ukraine
        </a>
        .
      </div>
    </div>
  );
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    const Description = () => (
      <Block background="light">
        {[
          {
            content:
              'This is another description of how this project is useful',
            image: `${baseUrl}img/docusaurus.svg`,
            imageAlign: 'right',
            title: 'Description',
          },
        ]}
      </Block>
    );

    const pre = '```';

    const codeExample = `${pre}python
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

# Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
sphere_mesh = ico_sphere(level=3)
verts, faces, _ = load_obj("model.obj")
test_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

# Differentiably sample 5k points from the surface of each mesh and then compute the loss.
sample_sphere = sample_points_from_meshes(sphere_mesh, 5000)
sample_test = sample_points_from_meshes(test_mesh, 5000)
loss_chamfer, _ = chamfer_distance(sample_sphere, sample_test)
    `;

    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <Container>
          <ol>
            <li>
              <strong>Install PyTorch3D  </strong> (following the instructions <a href="https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md">here</a>)
            </li>
            <li>
              <strong>Try a few 3D operators  </strong>
              e.g. compute the chamfer loss between two meshes:
              <MarkdownBlock>{codeExample}</MarkdownBlock>
            </li>
          </ol>
        </Container>
      </div>
    );

    const Features = () => (
      <div className="productShowcaseSection" style={{textAlign: 'center'}}>
        <Block layout="fourColumn">
          {[
            {
              content:
                'Supports batching of 3D inputs of different sizes ' +
                'such as meshes' ,
              image: `${baseUrl}img/batching.svg`,
              imageAlign: 'top',
              title: 'Heterogeneous Batching',
            },
            {
              content:
                'Supports optimized implementations of ' +
                'several common  functions for 3D data',
              image: `${baseUrl}img/ops.png`,
              imageAlign: 'top',
              title: 'Fast 3D Operators',
            },
            {
              content:
                'Modular differentiable rendering API ' +
                'with parallel implementations in ' +
                'PyTorch, C++ and CUDA' ,
              image: `${baseUrl}img/rendering.svg`,
              imageAlign: 'top',
              title: 'Differentiable Rendering',
            },
          ]}
        </Block>
      </div>
    );

    const Showcase = () => {
      if ((siteConfig.users || []).length === 0) {
        return null;
      }

      const showcase = siteConfig.users
        .filter(user => user.pinned)
        .map(user => (
          <a href={user.infoLink} key={user.infoLink}>
            <img src={user.image} alt={user.caption} title={user.caption} />
          </a>
        ));

      const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;

      return (
        <div className="productShowcaseSection paddingBottom">
          <h2>Who is Using This?</h2>
          <p>This project is used by all these people</p>
          <div className="logos">{showcase}</div>
          <div className="more-users">
            <a className="button" href={pageUrl('users.html')}>
              More {siteConfig.title} Users
            </a>
          </div>
        </div>
      );
    };

    return (
      <div>
        <SocialBanner />
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="landingPage mainContainer">
          <Features />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;
