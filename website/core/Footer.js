/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 */
const PropTypes = require("prop-types");
const React = require('react');

function SocialFooter(props) {
  const repoUrl = `https://github.com/${props.config.organizationName}/${props.config.projectName}`;
  return (
    <div className="footerSection">
      <div className="social">
        <a
          className="github-button" // part of the https://buttons.github.io/buttons.js script in siteConfig.js
          href={repoUrl}
          data-count-href={`${repoUrl}/stargazers`}
          data-show-count="true"
          data-count-aria-label="# stargazers on GitHub"
          aria-label="Star PyTorch3D on GitHub"
        >
          {props.config.projectName}
        </a>
      </div>
    </div>
  );
}

SocialFooter.propTypes = {
  config: PropTypes.object
};

class Footer extends React.Component {
  docUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    const docsUrl = this.props.config.docsUrl;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    return `${baseUrl}${docsPart}${langPart}${doc}`;
  }

  pageUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    return baseUrl + (language ? `${language}/` : '') + doc;
  }

  render() {
    const repoUrl = `https://github.com/${this.props.config.organizationName}/${this.props.config.projectName}`;
    return (
      <footer className="nav-footer" id="footer">
        <section className="sitemap">
          <SocialFooter config={this.props.config} />
        </section>

        <a
          href="https://opensource.facebook.com/"
          target="_blank"
          rel="noreferrer noopener"
          className="fbOpenSource">
          <img
            src={`${this.props.config.baseUrl}img/oss_logo.png`}
            alt="Facebook Open Source"
            width="170"
            height="45"
          />
        </a>
        <section className="copyright">{this.props.config.copyright}</section>
      </footer>
    );
  }
}

module.exports = Footer;
