---
layout: page
---

<script setup>
import {
  VPTeamPage,
  VPTeamPageTitle,
  VPTeamMembers,
  VPTeamPageSection
} from 'vitepress/theme'

// SVG icons
let svg_icons = new Object();

// Add stringified SVGs
svg_icons["url"] = '<svg viewBox="0 0 192 192" xmlns="http://www.w3.org/2000/svg" style="enable-background:new 0 0 192 192" xml:space="preserve" fill="#000000"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><path d="M84 128.6H54.6C36.6 128.6 22 114 22 96c0-9 3.7-17.2 9.6-23.1 5.9-5.9 14.1-9.6 23.1-9.6H84m24 65.3h29.4c9 0 17.2-3.7 23.1-9.6 5.9-5.9 9.6-14.1 9.6-23.1 0-18-14.6-32.6-32.6-32.6H108M67.9 96h56.2" style="fill:none;stroke:#707070;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10"></path></g></svg>'

svg_icons["rg"] = '<svg fill="#000000" viewBox="0 0 24 24" role="img" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><title>ResearchGate icon</title><path d="M19.586 0c-.818 0-1.508.19-2.073.565-.563.377-.97.936-1.213 1.68a3.193 3.193 0 0 0-.112.437 8.365 8.365 0 0 0-.078.53 9 9 0 0 0-.05.727c-.01.282-.013.621-.013 1.016a31.121 31.123 0 0 0 .014 1.017 9 9 0 0 0 .05.727 7.946 7.946 0 0 0 .077.53h-.005a3.334 3.334 0 0 0 .113.438c.245.743.65 1.303 1.214 1.68.565.376 1.256.564 2.075.564.8 0 1.536-.213 2.105-.603.57-.39.94-.916 1.175-1.65.076-.235.135-.558.177-.93a10.9 10.9 0 0 0 .043-1.207v-.82c0-.095-.047-.142-.14-.142h-3.064c-.094 0-.14.047-.14.141v.956c0 .094.046.14.14.14h1.666c.056 0 .084.03.084.086 0 .36 0 .62-.036.865-.038.244-.1.447-.147.606-.108.385-.348.664-.638.876-.29.212-.738.35-1.227.35-.545 0-.901-.15-1.21-.353-.306-.203-.517-.454-.67-.915a3.136 3.136 0 0 1-.147-.762 17.366 17.367 0 0 1-.034-.656c-.01-.26-.014-.572-.014-.939a26.401 26.403 0 0 1 .014-.938 15.821 15.822 0 0 1 .035-.656 3.19 3.19 0 0 1 .148-.76 1.89 1.89 0 0 1 .742-1.01c.344-.244.593-.352 1.137-.352.508 0 .815.096 1.144.303.33.207.528.492.764.925.047.094.111.118.198.07l1.044-.43c.075-.048.09-.115.042-.199a3.549 3.549 0 0 0-.466-.742 3 3 0 0 0-.679-.607 3.313 3.313 0 0 0-.903-.41A4.068 4.068 0 0 0 19.586 0zM8.217 5.836c-1.69 0-3.036.086-4.297.086-1.146 0-2.291 0-3.007-.029v.831l1.088.2c.744.144 1.174.488 1.174 2.264v11.288c0 1.777-.43 2.12-1.174 2.263l-1.088.2v.832c.773-.029 2.12-.086 3.465-.086 1.29 0 2.951.057 3.667.086v-.831l-1.49-.2c-.773-.115-1.174-.487-1.174-2.264v-4.784c.688.057 1.29.057 2.206.057 1.748 3.123 3.41 5.472 4.355 6.56.86 1.032 2.177 1.691 3.839 1.691.487 0 1.003-.086 1.318-.23v-.744c-1.031 0-2.063-.716-2.808-1.518-1.26-1.376-2.95-3.582-4.355-6.074 2.32-.545 4.04-2.722 4.04-4.9 0-3.208-2.492-4.698-5.758-4.698zm-.515 1.29c2.406 0 3.839 1.26 3.839 3.552 0 2.263-1.547 3.782-4.097 3.782-.974 0-1.404-.03-2.063-.086v-7.19c.66-.059 1.547-.059 2.32-.059z"></path></g></svg>'

svg_icons["gs"] = '<svg xmlns="http://www.w3.org/2000/svg" aria-label="Google Scholar" role="img" viewBox="0 0 512 512" fill="#000000"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><rect width="512" height="512" rx="15%" fill="#000000"></rect><path fill="#ffffff" d="M213 111l-107 94h69c5 45 41 64 78 67-7 18-4 27 7 39-43 1-103 26-103 67 4 45 63 54 92 54 38 1 81-19 90-54 4-35-10-54-31-71-23-18-28-28-21-40 15-17 35-27 39-51 2-17-2-28-6-43l45-38-1 16c-3 2-5 6-5 9v103c2 13 22 11 23 0V160c0-3-2-7-5-8v-25l16-16zm58 141c-61 10-87-87-38-99 56-11 83 86 38 99zm-5 73c60 13 61 63 10 78-44 9-82-4-81-30 0-25 35-48 71-48z"></path></g></svg>'

svg_icons["id"] = '<svg fill="#000000" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><path d="M 16 3 C 8.8321388 3 3 8.832144 3 16 C 3 23.167856 8.8321388 29 16 29 C 23.167861 29 29 23.167856 29 16 C 29 8.832144 23.167861 3 16 3 z M 16 5 C 22.086982 5 27 9.9130223 27 16 C 27 22.086978 22.086982 27 16 27 C 9.9130183 27 5 22.086978 5 16 C 5 9.9130223 9.9130183 5 16 5 z M 11 8 A 1 1 0 0 0 11 10 A 1 1 0 0 0 11 8 z M 10 11 L 10 22 L 12 22 L 12 11 L 10 11 z M 14 11 L 14 12 L 14 22 L 18.5 22 C 21.525577 22 24 19.525577 24 16.5 C 24 13.474423 21.525577 11 18.5 11 L 14 11 z M 16 13 L 18.5 13 C 20.444423 13 22 14.555577 22 16.5 C 22 18.444423 20.444423 20 18.5 20 L 16 20 L 16 13 z"></path></g></svg>'

svg_icons["tool"] = '<svg fill="#6c6c6c" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" stroke="#6c6c6c"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M-0.136 16.708c0.152-8.765 6.287-15.005 13.797-16.015 8.959-1.199 16.495 4.895 17.943 12.979 1.375 7.667-2.839 14.844-9.787 17.688-0.599 0.244-0.927 0.109-1.156-0.5l-3.453-8.969c-0.197-0.527-0.063-0.855 0.453-1.088 1.563-0.709 2.536-1.896 2.797-3.6 0.411-2.64-1.5-5.077-4.161-5.307-2.423-0.235-4.609 1.453-5 3.853-0.339 2.131 0.713 4.115 2.697 5.016 0.62 0.281 0.745 0.557 0.505 1.188l-3.469 9.031c-0.167 0.443-0.531 0.6-1 0.417-3.661-1.432-6.667-4.167-8.437-7.677-1.609-3.177-1.624-5.661-1.729-7.021zM1.213 16.584c0.027 0.427 0.041 0.921 0.084 1.427 0.405 4.64 3.197 9.26 8.452 11.817 0.209 0.093 0.287 0.052 0.365-0.145 0.959-2.527 1.927-5.052 2.901-7.579 0.083-0.208 0.041-0.307-0.152-0.427-2.041-1.287-3.057-3.131-2.943-5.552 0.063-1.391 0.6-2.615 1.537-3.636 1.932-2.109 4.968-2.568 7.453-1.135 2.052 1.187 3.197 3.484 2.916 5.839-0.235 1.968-1.244 3.479-2.953 4.5-0.172 0.104-0.224 0.187-0.145 0.389 0.979 2.532 1.953 5.063 2.916 7.595 0.079 0.203 0.157 0.244 0.36 0.145 2.297-1.068 4.208-2.599 5.688-4.64 2.244-3.115 3.171-6.579 2.728-10.391-0.88-7.584-7.703-13.865-16.489-12.781-6.844 0.839-12.604 6.615-12.719 14.573z"></path> </g></svg>'

svg_icons["dataset"] = '<svg viewBox="0 0 91 91" enable-background="new 0 0 91 91" id="Layer_1" version="1.1" xml:space="preserve" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" fill="#000000"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <g> <g> <path d="M88.587,25.783L46.581,5.826c-0.672-0.317-1.45-0.317-2.12,0L2.462,25.783 c-0.861,0.408-1.41,1.277-1.41,2.232c0,0.951,0.549,1.819,1.41,2.229l41.999,19.954c0.335,0.158,0.697,0.239,1.059,0.239 c0.363,0,0.726-0.081,1.061-0.239l42.006-19.954c0.861-0.41,1.41-1.278,1.41-2.229C89.997,27.06,89.448,26.191,88.587,25.783z" fill="#647F94"></path> <path d="M45.521,68.085c-0.483,0-0.965-0.105-1.414-0.317L2.109,47.813c-1.643-0.781-2.341-2.744-1.562-4.386 c0.78-1.642,2.742-2.341,4.388-1.562l40.584,19.283l40.595-19.283c1.639-0.78,3.606-0.083,4.386,1.562 c0.78,1.643,0.083,3.605-1.562,4.386L46.934,67.768C46.487,67.979,46.004,68.085,45.521,68.085z" fill="#45596B"></path> <path d="M45.521,84.912c-0.483,0-0.965-0.105-1.414-0.317L2.109,64.641c-1.643-0.78-2.341-2.746-1.562-4.389 c0.78-1.645,2.742-2.342,4.388-1.562l40.584,19.282L86.115,58.69c1.642-0.78,3.606-0.083,4.386,1.562 c0.78,1.643,0.083,3.608-1.56,4.389L46.934,84.595C46.487,84.807,46.004,84.912,45.521,84.912z" fill="#45596B"></path> </g> </g> </g></svg>'

// Authors
const authors = [
  {
    avatar: '../avatars/Gavin_Bosman.jpg',
    name: 'Gavin Bosman',
    title: 'Lead developer and maintainer',
    links: [
      { icon: 'linkedin', link: 'https://www.linkedin.com/in/gavin-bosman/' },
      { icon: {
            svg: svg_icons["id"]
        }, link: 'https://orcid.org/0009-0004-3349-0322' }
    ]
  }, {
    avatar: '../avatars/Livingstone_SR.png',
    name: 'Steven R. Livingstone',
    title: 'Project design and documentation',
    links: [
        { icon: {
          svg: svg_icons["url"]
        }, link: 'https://affectivedatascience.com/' },
        { icon: {
              svg: svg_icons["rg"]
        }, link: 'https://www.researchgate.net/profile/Steven-Livingstone' },
        {
          icon: {
              svg: svg_icons["gs"]
          }, link: 'https://scholar.google.com/citations?user=eEwWnkUAAAAJ'
        },
        { icon: {
              svg: svg_icons["id"]
        }, link: 'https://orcid.org/0000-0002-6364-6410' }
    ]
  }
]

</script>

<br>
<h1 style="text-align: center; font-size: 2.00rem; line-height: 3.5rem; font-weight: 375;">
  Authors
</h1>

<VPTeamPage>
  <!-- Header --> 
  <!-- <VPTeamPageTitle>
    <template #title>
      Authors
    </template>
  </VPTeamPageTitle> -->

  <!-- Current members -->
  <!-- <p id="authors"></p> -->
  <VPTeamMembers size="medium" :members="authors" />
<!--   
  <br>
  <br>
  <VPTeamPageSection>
    <template #title>PhD students</template>
    <template #members>
      <VPTeamMembers size="medium" :members="phd_students" />
    </template>
  </VPTeamPageSection> -->

</VPTeamPage>