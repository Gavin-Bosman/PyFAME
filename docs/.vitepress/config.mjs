import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "PyFAME Documentation",
  description: "API documentation for the PyFAME package",
  srcDir: './src',
  outDir: './docsite',
  base: '/PyFAME',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: './pyfame_logo.png',
    nav: [
      { text: 'Home', link: '/index' },
      { text: 'Reference', link: '/reference/overview' },
      { text: 'Examples', link: '/examples/examples' },
      { text: 'Changelog', link: '/extras/changelog'}
    ],

    footer: {
      message: 'Released under the <a href="https://github.com/Gavin-Bosman/PyFAME/blob/master/LICENSE.txt">Gnu General Public License Version 3</a>.',
      copyright: 'Copyright @ 2024-present <a href="https://github.com/Gavin-Bosman">Gavin Bosman</a>'
    },

    sidebar: {
      '/reference/': [
        {
          text: 'PyFAME API Reference',
          items: [
            { 
              text: 'API Overview', 
              link: '/reference/overview',
              items: [
                { text: 'Analysis Submodule', link: '/reference/overview#module_analysis' },
                { text: 'Coloring Submodule', link: '/reference/overview#module_coloring' },
                { text: 'Occlusion Submodule', link: '/reference/overview#module_occlusion' },
                { text: 'PLD Submodule', link: '/reference/overview#module_pld' },
                { text: 'Scrambling Submodule', link: '/reference/overview#module_scrambling' },
                { text: 'Temporal_transforms Submodule', link:'/reference/overview#module_tt' },
                { text: 'Utilities', link: '/reference/overview#module_utils' }
              ]
            },
            {
              text: 'Occlusion',
              link: '/reference/occlusion',
              items: [
                { text: 'Facial Masking', link: '/reference/occlusion#facial_masking' },
                { text: 'Facial Occlusion', link: '/reference/occlusion#facial_occlusion' },
                { text: 'Facial Blurring', link: '/reference/occlusion#facial_blurring' },
                { text: 'Applying Noise', link: '/reference/occlusion#facial_noise'}
              ]
            },
            {
              text: 'Coloring',
              link: '/reference/coloring',
              items: [
                { text: 'face_color_shift()', link: '/reference/coloring#face_color_shift' },
                { text: 'face_saturation_shift()', link: '/reference/coloring#face_sat_shift' },
                { text: 'face_brightness_shift()', link: '/reference/coloring#face_bright_shift' },
              ]
            }
          ]
        }
      ],

      '/examples/': [
        {
          text: 'Examples and Tutorials', 
          items: [
            { 
              text: 'Getting Started', 
              link: '/examples/examples#start',
              items: [
                { text: "Installation", link: '/examples/examples#install' },
                { text: "Using PyFAME", link: '/examples/examples#using_pyfame' }
              ] 
            },
            {text: 'Examples', link: '/examples/examples#examples'}
          ]
        }
      ]

    },
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Gavin-Bosman/PyFAME' }
    ]
  }
})
