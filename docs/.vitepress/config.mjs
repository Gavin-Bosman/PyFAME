import { withMermaid } from 'vitepress-plugin-mermaid'

// https://vitepress.dev/reference/site-config
export default withMermaid({
  title: "PyFAME",
  description: "API documentation for the PyFAME package",
  srcDir: './src',
  outDir: './docsite',
  base: '/PyFAME',

  mermaid: {
    securityLevel: 'loose',
    theme: 'default',
    layout: 'elk',
    themeVariables: {
      mergeEdges: true,
      nodePlacementStrategy: 'BRANDES_KOEPF',
    }
  },

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: './pyfame_logo.png',
    nav: [
      { text: 'Guide', link: '/guide/intro' },
      { text: 'Reference', link: '/reference/overview' },
      { text: 'About', link: '/about/authors' }
    ],

    footer: {
      message: 'This documentation is released under a <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC4.0 License</a>.',
      copyright: 'Copyright @ 2024-present <a href="https://github.com/Gavin-Bosman">Gavin Bosman</a>'
    },

    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'What Is PyFAME?', link: '/guide/intro' },
          { text: 'Getting Started', link: '/guide/getting_started' },
          { text: 'Examples', link: '/guide/examples' }
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'Overview', link: '/reference/overview' },
          { text: 'Moviefy', link: '/reference/moviefy' },
          { text: 'Analysis', link: '/reference/analysis' },
          { text: 'Coloring', link: '/reference/coloring' },
          { text: 'Occlusion', link: '/reference/occlusion' },
          { text: 'Point-Light Display', link: '/reference/pld' },
          { text: 'Scrambling', link: '/reference/scrambling' },
          { text: 'Temporal Transforms', link: '/reference/temporal_transforms' },
          { text: 'Utilities', link: '/reference/utils' },
          { text: 'Codebook', link: '/reference/codebook'}
        ]
      },
      {
        text: 'About',
        items: [
          { text: "Authors", link: '/about/authors' },
          { text: "Contributing", link: '/about/contrib'},
          { text: "Changelog", link: '/about/changelog' }
        ]
      }
    ],
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Gavin-Bosman/PyFAME' }
    ]
  }
})
