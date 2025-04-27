import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "PyFAME",
  description: "API documentation for the PyFAME package",
  srcDir: './src',
  outDir: './docsite',
  base: '/PyFAME',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: './pyfame_logo.png',
    nav: [
      { text: 'Reference', link: '/reference/overview' },
      { text: 'Guide', link: '/guide/getting_started' },
      { text: 'About', link: '/about/authors' }
    ],

    footer: {
      message: 'This documentation is released under a <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC4.0 License</a>.',
      copyright: 'Copyright @ 2024-present <a href="https://github.com/Gavin-Bosman">Gavin Bosman</a>'
    },

    sidebar: {
      '/reference/': [
        {
          text: 'Reference',
          items: [
            { 
              text: 'Overview', 
              link: '/reference/overview'
            },
            {
              text: 'Analysis',
              link: '/reference/analysis',
              items: [
                { text: 'Optical Flow', link: '/reference/analysis#optic_flow' },
                { text: 'Extracting Color Means', link: '/reference/analysis#color_means' }
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
              text: 'Point-Light Display',
              link: '/reference/pld'
            },
            {
              text: 'Scrambling',
              link: '/reference/scrambling'
            },
            {
              text: 'Temporal Transforms',
              link: '/reference/temporal_transforms',
              items: [
                {text: 'Shuffle Frame Order', link: '/reference/temporal_transforms#shuffle_frames'},
                {text: 'Generate Frame-Block Ordering', link: '/reference/temporal_transforms#frame_order'}
              ]
            },
            {
              text: 'Utilities',
              link: '/reference/utils',
              items: [
                {text: 'General Utilities', link: '/reference/utils#gen_utils'},
                {text: 'Transcoding Videos', link: '/reference/utils#transcode'},
                {text: 'Custom Landmark Paths', link: '/reference/utils#create_path'}
              ]
            }
          ]
        }
      ],

      '/guide/': [
        {
          text: 'Guide', 
          items: [
            { 
              text: 'Getting Started', 
              link: '/guide/getting_started',
              items: [
                { text: "Installation", link: '/guide/getting_started#install' },
                { text: "Quick Example", link: '/guide/getting_started#quick_example' }
              ] 
            },
            {text: 'Examples', link: '/guide/examples'}
          ]
        }
      ],

      '/about/': [
        {
          text: 'About', 
          items: [
            { text: "Authors", link: '/about/authors' },
            { text: "Changelog", link: '/about/changelog' }
          ]
        }
      ]

    },
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Gavin-Bosman/PyFAME' }
    ]
  }
})
