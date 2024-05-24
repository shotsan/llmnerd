# Hextra Starter Template

[![Deploy Hugo site to Pages](https://github.com/imfing/hextra-starter-template/actions/workflows/pages.yaml/badge.svg)](https://github.com/imfing/hextra-starter-template/actions/workflows/pages.yaml)
[![Netlify Status](https://api.netlify.com/api/v1/badges/6e83fd88-5ffe-4808-9689-c0f3b100bfe3/deploy-status)](https://app.netlify.com/sites/hextra-starter-template/deploys)



🐣 Minimal template for getting started with [Hextra](https://github.com/imfing/hextra)

![hextra-template](https://github.com/imfing/hextra-starter-template/assets/5097752/c403b9a9-a76c-47a6-8466-513d772ef0b7)

[🌐 Demo ↗](https://imfing.github.io/hextra-starter-template/)

## Quick Start

 Root folder for this page is content folder, so path of content / 
 add folders inside content and give them /folder_name as the link. 

 ## to be added
 1.css 
 2.java script
 3. comments section


## Deployment


### Netlify

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/imfing/hextra-starter-template)



## Local Development

Pre-requisites: [Hugo](https://gohugo.io/getting-started/installing/), [Go](https://golang.org/doc/install) and [Git](https://git-scm.com)

```shell
# Clone the repo
git clone https://github.com/imfing/hextra-starter-template.git

# Change directory
cd hextra-starter-template

# Start the server
hugo mod tidy
hugo server --logLevel debug --disableFastRender -p 1313
```

### Update theme

```shell
hugo mod get -u
hugo mod tidy
```

See [Update modules](https://gohugo.io/hugo-modules/use-modules/#update-modules) for more details.

